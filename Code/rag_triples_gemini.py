"""
rag_triples_gemini.py

RAG over Triples.csv (local “database”) + Gemini API for generation.

What it does:
1) Loads Triples.csv (Subject, Relationship, Object)
2) Builds “documents” per Subject from its triples
3) Embeds each Subject-doc with SentenceTransformer (local)
4) Builds FAISS index (local) + caches embeddings to disk
5) On each question: retrieves top-K subjects, builds a compact Context
6) Sends ONLY that Context + question to Gemini for a short answer

Requirements (run inside your venv):
  python -m pip install -U pandas numpy faiss-cpu sentence-transformers google-genai

Set your API key (PowerShell):
  setx GEMINI_API_KEY "YOUR_KEY_HERE"
Then restart terminal.

Place this file in the same folder as Triples.csv
(or set TRIPLES_CSV path below).
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from google import genai


# =========================
# 1) CONFIG
# =========================
TRIPLES_CSV = "Triples.csv"  # put Triples.csv in same folder as this script OR change to full path

# Column names in Triples.csv (case-sensitive). Change if your file uses different headers.
SUBJ_COL = "Subject"
REL_COL = "Relationship"
OBJ_COL = "Object"

# Embedding model (local)
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Gemini model (API)
GEMINI_MODEL = "gemini-2.5-flash"  # or "gemini-2.5-flash-lite" / "gemini-2.5-pro"

# Retrieval & context limits
TOP_K = 5                     # how many subjects to retrieve
MAX_FACTS_PER_SUBJECT = 120   # cap per subject so one subject doesn't dominate
MAX_CONTEXT_CHARS = 9000      # cap prompt size for speed/cost
MIN_SCORE_TO_SHOW = 0.0       # keep all, or set e.g. 0.2

# Cache directory (saves embeddings + index so you don’t recompute each run)
CACHE_DIR = Path(".rag_cache_triples")
CACHE_DIR.mkdir(exist_ok=True)


SYSTEM_PROMPT = (
    "Answer using ONLY the provided Context extracted from Triples.csv.\n"
    "If the Context does not contain the answer, reply exactly:\n"
    "Not found in the dataset.\n"
    "Keep the answer short and practical.\n"
)


# =========================
# 2) HELPERS
# =========================
def file_fingerprint(path: Path) -> str:
    """Fast-ish fingerprint to invalidate cache when Triples.csv changes."""
    stat = path.stat()
    base = f"{path.name}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def normalize_text(s: str) -> str:
    s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.strip()


def extract_location_hints(question: str) -> Dict[str, List[str]]:
    """
    Very lightweight heuristics to reduce ‘Irvine -> Riverside’ type errors.
    We try to detect:
      - CA city-like words the user typed (capitalized tokens)
      - ZIP codes (5 digits)
    Then we can prefer retrieved docs containing these tokens.
    """
    q = question.strip()

    zips = re.findall(r"\b(\d{5})\b", q)

    # City candidates: words with letters, not too short, not common question words
    common = {
        "what", "where", "which", "find", "nearest", "closest", "food", "pantry",
        "shelter", "services", "service", "in", "near", "around", "to", "the",
        "a", "an", "of", "for", "and", "is", "are", "i", "im", "me", "my"
    }
    tokens = re.findall(r"[A-Za-z]+", q)
    # keep tokens that look like proper nouns OR are known city tokens user typed in caps
    city_like = []
    for t in tokens:
        low = t.lower()
        if low in common:
            continue
        if len(t) >= 4 and (t[0].isupper() or t.isupper()):
            city_like.append(t)
    # also keep a direct mention of "irvine" etc even if not capitalized
    # (if user typed lowercase)
    lower_tokens = set(t.lower() for t in tokens)
    if "irvine" in lower_tokens and "Irvine" not in city_like:
        city_like.append("Irvine")

    return {"zips": zips, "cities": list(dict.fromkeys(city_like))}


def subject_doc_from_group(df_subj: pd.DataFrame) -> str:
    """
    Build a “document” string for one Subject from its triples.
    This doc is what we embed and what we show in Context.
    """
    # Keep consistent ordering and limit
    rows = df_subj.head(MAX_FACTS_PER_SUBJECT)
    lines = []
    subj_val = rows[SUBJ_COL].iloc[0]
    lines.append(f"Subject: {subj_val}")
    for _, r in rows.iterrows():
        rel = normalize_text(r.get(REL_COL, ""))
        obj = normalize_text(r.get(OBJ_COL, ""))
        if rel and obj:
            lines.append(f"{rel}: {obj}")
        elif rel:
            lines.append(f"{rel}:")
        elif obj:
            lines.append(f"- {obj}")
    return "\n".join(lines)


def load_triples(triples_path: Path) -> pd.DataFrame:
    df = pd.read_csv(triples_path)
    for col in [SUBJ_COL, REL_COL, OBJ_COL]:
        if col not in df.columns:
            raise ValueError(
                f"Triples.csv is missing column '{col}'. "
                f"Found columns: {list(df.columns)}. "
                f"Update SUBJ_COL/REL_COL/OBJ_COL at top of the script."
            )
    df = df[[SUBJ_COL, REL_COL, OBJ_COL]].fillna("")
    return df


# =========================
# 3) BUILD / LOAD INDEX
# =========================
def build_subject_docs(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Returns:
      subjects: list of unique subject ids (strings)
      docs:     list of doc strings per subject (same length)
    """
    subjects = []
    docs = []

    # groupby preserves sort order unless sort=True; we want stable output
    for subj, g in df.groupby(SUBJ_COL, sort=False):
        g = g.reset_index(drop=True)
        doc = subject_doc_from_group(g)
        subjects.append(str(subj))
        docs.append(doc)

    return subjects, docs


def cache_paths(triples_fp: str, embed_model_id: str) -> Dict[str, Path]:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", embed_model_id)
    prefix = f"{triples_fp}__{safe_model}"
    return {
        "subjects": CACHE_DIR / f"{prefix}__subjects.json",
        "docs": CACHE_DIR / f"{prefix}__docs.json",
        "emb": CACHE_DIR / f"{prefix}__emb.npy",
        "index": CACHE_DIR / f"{prefix}__faiss.index",
    }


def load_or_build_index(triples_path: Path, embedder: SentenceTransformer):
    triples_fp = file_fingerprint(triples_path)
    paths = cache_paths(triples_fp, EMBED_MODEL_ID)

    if all(p.exists() for p in paths.values()):
        subjects = json.loads(paths["subjects"].read_text(encoding="utf-8"))
        docs = json.loads(paths["docs"].read_text(encoding="utf-8"))
        emb = np.load(paths["emb"])
        index = faiss.read_index(str(paths["index"]))
        return subjects, docs, emb, index

    print("Building index (first time)…")
    df = load_triples(triples_path)
    subjects, docs = build_subject_docs(df)

    # Embed docs (batch)
    emb = embedder.encode(
        docs,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # important for cosine similarity via inner product
    ).astype("float32")

    # FAISS index: inner product (cosine because vectors normalized)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    # Save cache
    paths["subjects"].write_text(json.dumps(subjects, ensure_ascii=False), encoding="utf-8")
    paths["docs"].write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    np.save(paths["emb"], emb)
    faiss.write_index(index, str(paths["index"]))

    return subjects, docs, emb, index


# =========================
# 4) RETRIEVAL
# =========================
def retrieve(
    question: str,
    embedder: SentenceTransformer,
    subjects: List[str],
    docs: List[str],
    index: faiss.Index,
    top_k: int = TOP_K,
) -> List[Tuple[int, float]]:
    """
    Returns list of (doc_index, score) sorted by score desc.
    Score is cosine similarity (0..1-ish) because we normalize embeddings and use inner product.
    """
    q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, top_k)
    results = []
    for i, s in zip(idxs[0].tolist(), scores[0].tolist()):
        if i == -1:
            continue
        if s < MIN_SCORE_TO_SHOW:
            continue
        results.append((i, float(s)))
    return results


def apply_location_preference(
    question: str,
    candidates: List[Tuple[int, float]],
    docs: List[str],
) -> List[Tuple[int, float]]:
    """
    If user mentions a city or zip, prefer candidates that contain it.
    (This helps avoid ‘Irvine’ question returning ‘Riverside’ results.)
    """
    hints = extract_location_hints(question)
    if not hints["cities"] and not hints["zips"]:
        return candidates

    cities = hints["cities"]
    zips = hints["zips"]

    def doc_has_hint(doc: str) -> bool:
        dlow = doc.lower()
        for c in cities:
            if c.lower() in dlow:
                return True
        for z in zips:
            if z in doc:
                return True
        return False

    preferred = []
    others = []
    for i, s in candidates:
        (preferred if doc_has_hint(docs[i]) else others).append((i, s))

    # If we found any preferred matches, return them first
    return preferred + others


def build_context(candidates: List[Tuple[int, float]], subjects: List[str], docs: List[str]) -> str:
    chunks = []
    total = 0
    for rank, (i, score) in enumerate(candidates, start=1):
        header = f"[Match {rank} | score={score:.3f}]\n"
        block = header + docs[i].strip() + "\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        chunks.append(block)
        total += len(block)

    return "\n---\n".join(chunks).strip()


# =========================
# 5) GEMINI CALL
# =========================
def gemini_client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. In PowerShell run:\n"
            '  setx GEMINI_API_KEY "YOUR_KEY"\n'
            "Then restart VS Code terminal."
        )
    return genai.Client(api_key=key)


def call_gemini(client: genai.Client, system_prompt: str, user_prompt: str) -> str:
    # Keep it simple: pack system + user into one message.
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
        ],
    )
    return (resp.text or "").strip()


def answer_question(
    question: str,
    embedder: SentenceTransformer,
    client: genai.Client,
    subjects: List[str],
    docs: List[str],
    index: faiss.Index,
) -> str:
    # 1) retrieve
    candidates = retrieve(question, embedder, subjects, docs, index, top_k=TOP_K)
    candidates = apply_location_preference(question, candidates, docs)

    # 2) context
    context = build_context(candidates, subjects, docs)
    if not context:
        return "Not found in the dataset."

    # 3) ask Gemini
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
    return call_gemini(client, SYSTEM_PROMPT, user_prompt)


# =========================
# 6) MAIN LOOP
# =========================
def main():
    triples_path = Path(TRIPLES_CSV)
    if not triples_path.exists():
        # Try “Code/Triples.csv” if user runs from repo root
        alt = Path("Code") / TRIPLES_CSV
        if alt.exists():
            triples_path = alt
        else:
            raise FileNotFoundError(
                f"Cannot find {TRIPLES_CSV}.\n"
                f"Looked in: {Path.cwd() / TRIPLES_CSV}\n"
                f"Also tried: {Path.cwd() / alt}\n"
                "Fix TRIPLES_CSV path at the top of the script."
            )

    print(f"Using Triples file: {triples_path.resolve()}")

    print(f"Loading embedder: {EMBED_MODEL_ID}")
    embedder = SentenceTransformer(EMBED_MODEL_ID)

    subjects, docs, emb, index = load_or_build_index(triples_path, embedder)
    print(f"Index ready. Subjects: {len(subjects)}")

    client = gemini_client()
    print("\nDataset-grounded assistant ready.")
    print("Type a question. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        t0 = time.time()
        try:
            ans = answer_question(q, embedder, client, subjects, docs, index)
            dt = time.time() - t0
            print(f"\nAssistant ({dt:.2f}s):\n{ans}\n")
        except Exception as e:
            print("\nError:\n", repr(e), "\n")


if __name__ == "__main__":
    main()