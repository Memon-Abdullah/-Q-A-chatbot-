# langgraph_rag.py — CLEAN VERSION for fast api
# =====================================

import os
import pickle
import json
import numpy as np
import faiss
import requests
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBED_MODEL        = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
LLM_MODEL          = "openai/gpt-oss-120b:free"
TOP_K              = 5

SYSTEM_PROMPT = """You are a helpful assistant. You are given CONTEXT extracted from a Deep Learning textbook.
Answer the user's question using ONLY the provided context.
If the answer is not in the context, say: "I couldn't find this in the document."
Be concise, clear, and accurate."""

# ── State ────────────────────────────────────────────
class RAGState(TypedDict):
    question:         str
    query_vector:     list
    retrieved_chunks: list[str]
    answer:           str

# ── FAISS Load ───────────────────────────────────────
# Ye function sirf main.py call karega — yahan koi global load NAHI hai
def load_index(directory: str):
    faiss_path  = os.path.join(directory, "index.faiss")
    chunks_path = os.path.join(directory, "chunks.pkl")

    if not os.path.exists(faiss_path):
        raise FileNotFoundError("FAISS index nahi mila. Pehle 1_create_vectordb.py chalao.")

    index = faiss.read_index(faiss_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"[LOAD] {index.ntotal} vectors, {len(chunks)} chunks ready.")
    return index, chunks

# ── Node 1 ───────────────────────────────────────────
def embed_query_node(state: RAGState) -> dict:
    print("[Node 1] Embedding question...")

    response = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": EMBED_MODEL,
            "input": [state["question"]],
        },
        timeout=30
    )

    if response.status_code != 200:
        raise RuntimeError(f"Embedding error {response.status_code}: {response.text}")

    vector = response.json()["data"][0]["embedding"]
    return {"query_vector": vector}

# ── Node 2 ───────────────────────────────────────────
# faiss_index aur chunks bahar se aate hain — closure ke through
def make_retrieve_node(faiss_index, chunks):
    def retrieve_node(state: RAGState) -> dict:
        print(f"[Node 2] FAISS search top {TOP_K}...")
        vector = np.array(state["query_vector"], dtype=np.float32).reshape(1, -1)
        distances, indices = faiss_index.search(vector, TOP_K)

        retrieved = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                retrieved.append(chunks[idx])

        print(f"[Node 2] {len(retrieved)} chunks mile.")
        return {"retrieved_chunks": retrieved}
    
    return retrieve_node  # function return ho raha hai

# ── Node 3 ───────────────────────────────────────────
def generate_node(state: RAGState) -> dict:
    print("[Node 3] LLM call kar raha hun...")

    context_block = "\n\n---\n\n".join(
        [f"[Chunk {i+1}]:\n{chunk}" for i, chunk in enumerate(state["retrieved_chunks"])]
    )

    user_prompt = (
        f"CONTEXT FROM DOCUMENT:\n{'='*50}\n"
        f"{context_block}\n{'='*50}\n\n"
        f"USER QUESTION:\n{state['question']}\n\n"
        f"Please answer based on the context above."
    )

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": False,
        },
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(f"LLM error {response.status_code}: {response.text}")

    answer = response.json()["choices"][0]["message"]["content"].strip()
    return {"answer": answer}

# ── Graph Builder ─────────────────────────────────────
def build_rag_graph(faiss_index, chunks):
    graph = StateGraph(RAGState)

    graph.add_node("embed",    embed_query_node)
    graph.add_node("retrieve", make_retrieve_node(faiss_index, chunks))
    graph.add_node("generate", generate_node)

    graph.set_entry_point("embed")
    graph.add_edge("embed",    "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()