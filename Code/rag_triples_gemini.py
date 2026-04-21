"""
rag_triples_gemini.py

Stable presentation version:
- Local CPU retrieval over Triples.csv
- Optional Gemini generation
- Safe local fallback if Gemini is unavailable

Requirements:
  python -m pip install -U pandas numpy faiss-cpu sentence-transformers google-genai

Optional API key for Gemini:
  setx GEMINI_API_KEY "YOUR_KEY"
Then restart terminal.
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

try:
    from google import genai
except Exception:
    genai = None


# =========================
# 1) CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent

# Safer path handling
TRIPLES_CSV = BASE_DIR / "Triples.csv"
if not TRIPLES_CSV.exists():
    alt_csv = BASE_DIR.parent / "Triples.csv"
    if alt_csv.exists():
        TRIPLES_CSV = alt_csv

SUBJ_COL = "Subject"
REL_COL = "Relationship"
OBJ_COL = "Object"

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Presentation-safe default:
# False = local-only mode, no API dependence
USE_GEMINI = False

# If you want to try Gemini later, change USE_GEMINI to True
GEMINI_MODEL = "gemini-2.5-flash-lite"

TOP_K = 5
MAX_FACTS_PER_SUBJECT = 120
MAX_CONTEXT_CHARS = 7000
MIN_SCORE_TO_SHOW = 0.15

CACHE_DIR = BASE_DIR / ".rag_cache_triples"
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
    stat = path.stat()
    base = f"{path.name}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def normalize_text(s: str) -> str:
    s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(s.strip().split())


def extract_location_hints(question: str) -> Dict[str, List[str]]:
    q = question.strip()

    zips = re.findall(r"\b(\d{5})\b", q)

    common = {
        "what", "where", "which", "find", "nearest", "closest", "food", "pantry",
        "shelter", "services", "service", "in", "near", "around", "to", "the",
        "a", "an", "of", "for", "and", "is", "are", "i", "im", "me", "my",
        "need", "help", "with", "from", "at"
    }

    tokens = re.findall(r"[A-Za-z]+", q)
    city_like = []

    for t in tokens:
        low = t.lower()
        if low in common:
            continue
        if len(t) >= 4 and (t[0].isupper() or t.isupper()):
            city_like.append(t)

    lower_tokens = set(t.lower() for t in tokens)
    known_city_candidates = [
        "irvine", "riverside", "los", "angeles", "anaheim", "pomona",
        "ontario", "pasadena", "fullerton", "santa", "ana", "long", "beach"
    ]
    for city in known_city_candidates:
        if city in lower_tokens and city.capitalize() not in city_like:
            city_like.append(city.capitalize())

    return {"zips": zips, "cities": list(dict.fromkeys(city_like))}


def subject_doc_from_group(df_subj: pd.DataFrame) -> str:
    rows = df_subj.head(MAX_FACTS_PER_SUBJECT)
    lines = []

    subj_val = normalize_text(rows[SUBJ_COL].iloc[0])
    lines.append(f"Subject: {subj_val}")

    seen = set()

    for _, r in rows.iterrows():
        rel = normalize_text(r.get(REL_COL, ""))
        obj = normalize_text(r.get(OBJ_COL, ""))

        if not rel and not obj:
            continue

        key = (rel.lower(), obj.lower())
        if key in seen:
            continue
        seen.add(key)

        if rel and obj:
            lines.append(f"{rel}: {obj}")
        elif rel:
            lines.append(f"{rel}:")
        elif obj:
            lines.append(f"- {obj}")

    return "\n".join(lines)


def load_triples(triples_path: Path) -> pd.DataFrame:
    if not triples_path.exists():
        raise FileNotFoundError(f"Cannot find Triples.csv at: {triples_path}")

    df = pd.read_csv(triples_path)

    for col in [SUBJ_COL, REL_COL, OBJ_COL]:
        if col not in df.columns:
            raise ValueError(
                f"Triples.csv is missing column '{col}'. "
                f"Found columns: {list(df.columns)}"
            )

    df = df[[SUBJ_COL, REL_COL, OBJ_COL]].fillna("")
    return df


# =========================
# 3) BUILD / LOAD INDEX
# =========================
def build_subject_docs(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    subjects = []
    docs = []

    for subj, g in df.groupby(SUBJ_COL, sort=False):
        g = g.reset_index(drop=True)
        doc = subject_doc_from_group(g)

        if not doc.strip():
            continue

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

    print("Building index (first time)...")
    df = load_triples(triples_path)
    subjects, docs = build_subject_docs(df)

    emb = embedder.encode(
        docs,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

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
    q_emb = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

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
        if doc_has_hint(docs[i]):
            preferred.append((i, s))
        else:
            others.append((i, s))

    return preferred + others


def build_context(candidates: List[Tuple[int, float]], subjects: List[str], docs: List[str]) -> str:
    chunks = []
    total = 0

    for rank, (i, score) in enumerate(candidates, start=1):
        block = f"[Match {rank} | score={score:.3f}]\n{docs[i].strip()}\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        chunks.append(block)
        total += len(block)

    return "\n---\n".join(chunks).strip()


def extract_field(text, field):
    import re
    pattern = rf"{field}:\s*(.*)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return "Not available"


def format_local_answer(candidates, subjects, docs):
    if not candidates:
        return "Not found in the dataset."

    lines = []
    lines.append("Top matching services:\n")

    for rank, (i, score) in enumerate(candidates[:5], start=1):

        doc = docs[i]

        service_name = subjects[i]

        address = extract_field(doc, "Location_Address")
        zipcode = extract_field(doc, "Zipcode")
        website = extract_field(doc, "Website")
        service_type = extract_field(doc, "Service_Type")

        monday = extract_field(doc, "Monday")
        tuesday = extract_field(doc, "Tuesday")
        wednesday = extract_field(doc, "Wednesday")
        thursday = extract_field(doc, "Thursday")

        hours = f"Mon–Thu {monday}"

        lines.append(
            f"{rank}. {service_name}\n"
            f"   Address: {address}\n"
            f"   Zipcode: {zipcode}\n"
            f"   Type: {service_type}\n"
            f"   Hours: {hours}\n"
            f"   Website: {website}\n"
        )

    return "\n".join(lines)


# =========================
# 5) GEMINI
# =========================
def gemini_client() -> Optional["genai.Client"]:
    if not USE_GEMINI:
        return None

    if genai is None:
        print("Gemini SDK not available. Using local-only mode.")
        return None

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY not found. Using local-only mode.")
        return None

    try:
        return genai.Client(api_key=key)
    except Exception:
        print("Failed to create Gemini client. Using local-only mode.")
        return None


def call_gemini_with_retry(client: "genai.Client", system_prompt: str, user_prompt: str) -> str:
    retries = 3
    wait_seconds = 4
    last_error = None

    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    {
                        "role": "user",
                        "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]
                    }
                ],
            )
            text = (resp.text or "").strip()
            if text:
                return text
            return "Not found in the dataset."
        except Exception as e:
            last_error = e
            err = str(e)
            if "503" in err or "UNAVAILABLE" in err:
                print(f"Gemini busy. Retry {attempt + 1}/{retries}...")
                time.sleep(wait_seconds)
            else:
                break

    raise last_error


def answer_question(
    question: str,
    embedder: SentenceTransformer,
    client: Optional["genai.Client"],
    subjects: List[str],
    docs: List[str],
    index: faiss.Index,
) -> str:
    candidates = retrieve(question, embedder, subjects, docs, index, top_k=TOP_K)
    candidates = apply_location_preference(question, candidates, docs)

    if not candidates:
        return "Not found in the dataset."

    context = build_context(candidates, subjects, docs)

    if client is None:
        return format_local_answer(candidates, subjects, docs)

    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"

    try:
        return call_gemini_with_retry(client, SYSTEM_PROMPT, user_prompt)
    except Exception:
        return format_local_answer(candidates, subjects, docs)


# =========================
# 6) MAIN
# =========================
def main():
    if not TRIPLES_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find Triples.csv.\nLooked in: {TRIPLES_CSV}"
        )

    print(f"Using Triples file: {TRIPLES_CSV.resolve()}")
    print(f"Loading embedder: {EMBED_MODEL_ID}")

    embedder = SentenceTransformer(EMBED_MODEL_ID)

    subjects, docs, emb, index = load_or_build_index(TRIPLES_CSV, embedder)
    print(f"Index ready. Subjects: {len(subjects)}")

    client = gemini_client()

    if client is None:
        print("\nRunning in LOCAL-ONLY mode.")
    else:
        print("\nRunning in GEMINI mode with local fallback.")

    print("Dataset-grounded assistant ready.")
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
        except KeyboardInterrupt:
            print("\nInterrupted.\n")
            break
        except Exception as e:
            print(f"\nError:\n{repr(e)}\n")


if __name__ == "__main__":
    main()