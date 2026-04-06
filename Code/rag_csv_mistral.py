import os
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# CONFIG
# =========================
TRIPLES_CSV = "Triples.csv"   # put Triples.csv in the same folder as this script
SUBJECT_COL = "Subject"
REL_COL = "Relationship"
OBJ_COL = "Object"

# Embedding model (fast + good)
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# If download issues, use:
# EMBED_MODEL_ID = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# LLM choices (CPU-friendly recommended)
# If you're on Windows CPU, Phi-3-mini is much faster than Mistral-7B
LLM_ID = "microsoft/Phi-3-mini-4k-instruct"
# If you really want Mistral:
# LLM_ID = "mistralai/Mistral-7B-Instruct-v0.3"

TOP_K = 4                 # number of subjects to retrieve
MAX_FACTS_PER_SUBJECT = 120  # prevent one subject from becoming too huge
MAX_CONTEXT_CHARS = 8000  # cap prompt size for speed/stability
MAX_NEW_TOKENS = 180      # shorter = faster on CPU
DO_SAMPLE = False

SYSTEM_PROMPT = (
    "Answer using ONLY the provided Context from Triples.csv.\n"
    "If the Context does not contain the answer, reply exactly:\n"
    "Not found in the dataset.\n"
    "Keep the answer short and practical.\n"
)

# Where to cache embeddings so you don’t recompute every run
CACHE_DIR = Path(".rag_cache")
CACHE_DIR.mkdir(exist_ok=True)
EMB_FILE = CACHE_DIR / "triples_subject_embeddings.npy"
SUBJECTS_FILE = CACHE_DIR / "subjects.json"
DOCS_FILE = CACHE_DIR / "docs.json"
META_FILE = CACHE_DIR / "meta.json"


# =========================
# HELPERS
# =========================
def normalize_text(x: str) -> str:
    return " ".join(str(x).strip().split())


def build_subject_docs(triples_df: pd.DataFrame):
    """
    Convert triples into docs:
      one doc per Subject:
        Service: <Subject>
        <Relationship>: <Object>
        ...
    """
    triples_df = triples_df[[SUBJECT_COL, REL_COL, OBJ_COL]].fillna("")
    triples_df[SUBJECT_COL] = triples_df[SUBJECT_COL].map(normalize_text)
    triples_df[REL_COL] = triples_df[REL_COL].map(normalize_text)
    triples_df[OBJ_COL] = triples_df[OBJ_COL].map(normalize_text)

    docs = []
    subjects = []

    for subject, g in triples_df.groupby(SUBJECT_COL):
        # Deduplicate facts while preserving order
        seen = set()
        facts = []
        for _, row in g.iterrows():
            rel = row[REL_COL]
            obj = row[OBJ_COL]
            if not rel or not obj:
                continue
            key = (rel.lower(), obj.lower())
            if key in seen:
                continue
            seen.add(key)
            facts.append(f"{rel}: {obj}")
            if len(facts) >= MAX_FACTS_PER_SUBJECT:
                break

        doc = "Service: " + subject + "\n" + "\n".join(facts)
        subjects.append(subject)
        docs.append(doc)

    return subjects, docs


def load_or_build_index(triples_path: str, embedder: SentenceTransformer):
    """
    Loads cached embeddings/docs if CSV hasn't changed; otherwise rebuilds.
    """
    triples_path = Path(triples_path)
    if not triples_path.exists():
        raise FileNotFoundError(f"Cannot find {triples_path.resolve()}")

    csv_mtime = triples_path.stat().st_mtime

    if META_FILE.exists() and EMB_FILE.exists() and SUBJECTS_FILE.exists() and DOCS_FILE.exists():
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        if meta.get("csv_mtime") == csv_mtime and meta.get("embed_model") == EMBED_MODEL_ID:
            subjects = json.loads(SUBJECTS_FILE.read_text(encoding="utf-8"))
            docs = json.loads(DOCS_FILE.read_text(encoding="utf-8"))
            emb = np.load(EMB_FILE).astype(np.float32)
            return subjects, docs, emb

    # Rebuild
    df = pd.read_csv(triples_path)
    # Basic validation
    for c in [SUBJECT_COL, REL_COL, OBJ_COL]:
        if c not in df.columns:
            raise ValueError(f"Triples.csv missing required column: {c}. Found columns: {list(df.columns)}")

    print(f"Loaded {len(df)} triples from {triples_path.name}")
    print("Columns:", list(df.columns))

    subjects, docs = build_subject_docs(df)

    print(f"Built {len(subjects)} subject-docs. Embedding…")
    emb = embedder.encode(docs, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)

    # Cache
    SUBJECTS_FILE.write_text(json.dumps(subjects, ensure_ascii=False), encoding="utf-8")
    DOCS_FILE.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    np.save(EMB_FILE, emb)
    META_FILE.write_text(json.dumps({"csv_mtime": csv_mtime, "embed_model": EMBED_MODEL_ID}), encoding="utf-8")

    return subjects, docs, emb


def retrieve(query: str, subjects, docs, doc_emb, embedder, top_k: int = TOP_K):
    """
    Cosine similarity retrieval using dot product on normalized embeddings.
    """
    q_emb = embedder.encode([query], normalize_embeddings=True).astype(np.float32)[0]
    scores = doc_emb @ q_emb
    top_idx = np.argsort(-scores)[:top_k]

    results = []
    for idx in top_idx:
        results.append((float(scores[idx]), subjects[idx], docs[idx]))
    return results


def build_context(results):
    """
    Format retrieved docs into a context block and cap its size.
    """
    blocks = []
    for rank, (score, subj, doc) in enumerate(results, start=1):
        blocks.append(f"[Match {rank} | score={score:.3f}]\n{doc}")

    context = "\n\n---\n\n".join(blocks)

    # Hard cap to avoid extremely long prompts (speed + stability)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated]"
    return context


def load_llm(llm_id: str):
    tokenizer = AutoTokenizer.from_pretrained(llm_id)

    # Choose dtype safely for CPU vs GPU
    if torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        llm_id,
        device_map="auto",
        torch_dtype=dtype
    )
    model.eval()
    return tokenizer, model


def ask_llm(question: str, context: str, tokenizer, model) -> str:
    user_content = f"Context:\n{context}\n\nQuestion:\n{question}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # IMPORTANT: return_dict=True then generate(**inputs) avoids the 'shape' crash
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Attempt to strip prompt echoes (varies by model)
    # Keep only last assistant-ish segment if present
    if "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()

    return text.strip()


# =========================
# MAIN
# =========================
def main():
    print("Loading embedder:", EMBED_MODEL_ID)
    embedder = SentenceTransformer(EMBED_MODEL_ID)

    subjects, docs, doc_emb = load_or_build_index(TRIPLES_CSV, embedder)

    print("Loading LLM:", LLM_ID)
    tokenizer, model = load_llm(LLM_ID)

    print("\nDataset-grounded assistant ready (Triples.csv).")
    print("Ask questions. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        try:
            results = retrieve(q, subjects, docs, doc_emb, embedder, TOP_K)
            context = build_context(results)

            ans = ask_llm(q, context, tokenizer, model)
            print("\nAssistant:\n", ans, "\n")

        except KeyboardInterrupt:
            print("\nInterrupted.\n")
            break
        except Exception:
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()