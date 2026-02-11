import traceback
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ========= 1) CONFIG =========
CSV_PATH = r"services.csv"  # path to your CSV file

LLM_ID = "mistralai/Mistral-7B-Instruct-v0.3"
EMBED_ID = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 6          # how many rows to retrieve per question
MAX_NEW_TOKENS = 220

STRICT_SYSTEM = (
    "You are a helpful assistant that answers ONLY using the provided context from a dataset.\n"
    "Do NOT use any outside knowledge.\n"
    "If the answer is not explicitly in the context, respond exactly:\n"
    "Not found in the dataset.\n"
    "Keep the answer short and practical."
)

# ========= 2) LOAD CSV =========
df = pd.read_csv(CSV_PATH)

# Make a text representation of each row for retrieval + context
# This works even if we don't know the exact column names.
df = df.fillna("")
columns = df.columns.tolist()

def row_to_text(row) -> str:
    parts = []
    for c in columns:
        val = str(row[c]).strip()
        if val:
            parts.append(f"{c}: {val}")
    return "\n".join(parts)

docs = [row_to_text(df.iloc[i]) for i in range(len(df))]

print(f"Loaded {len(df)} rows from CSV.")
print("Columns:", columns)

# ========= 3) BUILD EMBEDDINGS + FAISS INDEX =========
embedder = SentenceTransformer(EMBED_ID)
doc_emb = embedder.encode(docs, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)

dim = doc_emb.shape[1]
index = faiss.IndexFlatIP(dim)  # cosine similarity because vectors are normalized
index.add(doc_emb)

# ========= 4) LOAD LLM =========
tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
model = AutoModelForCausalLM.from_pretrained(
    LLM_ID,
    device_map="auto",
    torch_dtype=torch.float16
)

def retrieve_context(question: str, k: int = TOP_K) -> str:
    q_emb = embedder.encode([question], normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q_emb, k)

    chunks = []
    for rank, idx in enumerate(ids[0]):
        if idx == -1:
            continue
        chunks.append(f"[Row {idx} | score={scores[0][rank]:.3f}]\n{docs[idx]}\n")
    return "\n---\n".join(chunks)

def ask_llm(question: str) -> str:
    context = retrieve_context(question, TOP_K)

    user_content = f"Context (dataset rows):\n{context}\n\nQuestion:\n{question}"

    messages = [
        {"role": "system", "content": STRICT_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()

    return text   # ✅ INSIDE the function


# ========= 5) SIMPLE CHAT LOOP =========
print("\nDataset-grounded assistant ready.")
print("Type a question. Type 'exit' to quit.\n")

while True:
    q = input("You: ").strip()
    if not q:
        continue
    if q.lower() in {"exit", "quit"}:
        break

    try:
        ans = ask_llm(q)
        print("\nAssistant:\n", ans, "\n")
    except Exception as e:
        traceback.print_exc()
        break