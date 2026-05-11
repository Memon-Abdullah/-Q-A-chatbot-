"""
FILE 1 — CREATE VECTOR DATABASE
================================
Flow:
  1. Read DLBook.docx  →  extract all text
  2. Split text into chunks
  3. Send each chunk to OpenRouter Embeddings API  (nvidia/llama-nemotron-embed-vl-1b-v2:free)
  4. Build FAISS index from embeddings
  5. Save FAISS index + raw chunks to disk  →  faiss_index/

Run this ONCE (or whenever your document changes).
"""

import os
import pickle
import time
import numpy as np
import faiss
import requests
from docx import Document
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBED_MODEL        = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
DOCX_PATH          = r"D:\AI\Rag_Project\DLBook.docx"
INDEX_DIR          = "faiss_index"
CHUNK_SIZE         = 500        # characters per chunk
CHUNK_OVERLAP      = 100        # overlap between consecutive chunks
EMBED_BATCH_SIZE   = 8          # how many chunks to embed per API call
DELAY_BETWEEN_BATCHES = 1.0     # seconds — avoids rate-limiting on free tier

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_docx(path: str) -> str:
    """Extract all paragraph text from a .docx file."""
    doc = Document(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)
    print(f"[1/4] Read {len(paragraphs)} paragraphs  ({len(full_text):,} characters) from '{path}'")
    return full_text


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding-window character-level chunker."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap          # slide forward with overlap
    print(f"[2/4] Created {len(chunks)} chunks  (size={chunk_size}, overlap={overlap})")
    return chunks


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Call OpenRouter /embeddings endpoint.
    Returns a list of float vectors (one per input text).
    """
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBED_MODEL,
        "input": texts,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(
            f"Embedding API error {response.status_code}: {response.text}"
        )

    data = response.json()
    # OpenAI-compatible response: data["data"] is a list of {index, embedding, object}
    embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
    return embeddings


def build_faiss_index(chunks: list[str]) -> faiss.IndexFlatL2:
    """Embed all chunks in batches and build a FAISS IndexFlatL2."""
    all_embeddings = []
    total_batches = (len(chunks) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    print(f"[3/4] Embedding {len(chunks)} chunks in {total_batches} batches ...")

    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        batch_num = i // EMBED_BATCH_SIZE + 1

        try:
            embeddings = get_embeddings(batch)
            all_embeddings.extend(embeddings)
            print(f"      Batch {batch_num}/{total_batches} done  ({len(embeddings)} vectors)")
        except Exception as e:
            print(f"      [ERROR] Batch {batch_num} failed: {e}")
            raise

        if i + EMBED_BATCH_SIZE < len(chunks):
            time.sleep(DELAY_BETWEEN_BATCHES)   # be polite to the free-tier API

    # Convert to numpy float32 matrix
    matrix = np.array(all_embeddings, dtype=np.float32)
    dim = matrix.shape[1]
    print(f"      Embedding dimension: {dim}")

    # Build FAISS index
    index = faiss.IndexFlatL2(dim)   # exact L2 distance search
    index.add(matrix)
    print(f"      FAISS index contains {index.ntotal} vectors")
    return index


def save_index(index: faiss.IndexFlatL2, chunks: list[str], directory: str):
    """Save FAISS index and raw chunks to disk."""
    os.makedirs(directory, exist_ok=True)

    faiss_path  = os.path.join(directory, "index.faiss")
    chunks_path = os.path.join(directory, "chunks.pkl")

    faiss.write_index(index, faiss_path)

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"[4/4] Saved FAISS index  →  {faiss_path}")
    print(f"      Saved chunks       →  {chunks_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  RAG Project — Step 1: Build Vector Database")
    print("=" * 55)

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found. Check your .env file.")

    # 1. Read document
    text = read_docx(DOCX_PATH)

    # 2. Chunk text
    chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)

    # 3. Embed & build FAISS index
    index = build_faiss_index(chunks)

    # 4. Save to disk
    save_index(index, chunks, INDEX_DIR)

    print("\n✅ Vector database created successfully!")
    print(f"   Location : {os.path.abspath(INDEX_DIR)}")
    print(f"   Chunks   : {len(chunks)}")
    print(f"   Vectors  : {index.ntotal}")


if __name__ == "__main__":
    main()
