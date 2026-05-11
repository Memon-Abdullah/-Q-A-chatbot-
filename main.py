# main.py
# =====================================================
# FastAPI Server — RAG Pipeline
# =====================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os

# Tumhara existing LangGraph file se import
from rag_pipeline_for_fast_api import build_rag_graph, load_index, RAGState

# ── Pydantic Models ─────────────────────────────────
# Request ka shape — Frontend se ye aayega
class QuestionRequest(BaseModel):
    question: str

# Response ka shape — Frontend ko ye jayega
class AnswerResponse(BaseModel):
    answer:           str
    chunks_retrieved: int
    question:         str

# ── Global Variables ─────────────────────────────────
rag_graph  = None   # Graph object
faiss_index = None
chunks_data = None

# ── Lifespan (startup + shutdown) ────────────────────
# @app.on_event("startup") purana tarika tha
# Naya tarika: lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP — server start hone par chalega
    global rag_graph, faiss_index, chunks_data
    print("[STARTUP] Loading FAISS index...")
    faiss_index, chunks_data = load_index("faiss_index")
    print("[STARTUP] Building LangGraph...")
    rag_graph = build_rag_graph(faiss_index, chunks_data)
    print("[STARTUP] Ready!")
    
    yield  # ← server yahan chalta rahega
    
    # SHUTDOWN — server band hone par chalega
    print("[SHUTDOWN] Cleaning up...")

# ── FastAPI App ──────────────────────────────────────
app = FastAPI(
    title="RAG API",
    description="Deep Learning Book Q&A via RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware — zaroori hai warna browser block karega
# HTML file directly open karo toh bhi kaam kare
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # production mein specific domain do
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── GET / — Health Check ─────────────────────────────
@app.get("/")
def health_check():
    """
    Server chal raha hai ya nahi check karo.
    Browser mein directly open karo: http://localhost:8000/
    """
    return {
        "status": "running",
        "model":  "openai/gpt-oss-120b:free",
        "index":  f"{faiss_index.ntotal} vectors loaded" if faiss_index else "not loaded"
    }

# ── GET /health — Detailed Status ───────────────────
@app.get("/health")
def detailed_health():
    return {
        "faiss_loaded":  faiss_index is not None,
        "chunks_count":  len(chunks_data) if chunks_data else 0,
        "graph_ready":   rag_graph is not None,
    }

# ── POST /ask — Main RAG Endpoint ───────────────────
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Frontend se question aata hai
    → LangGraph RAG pipeline chalti hai
    → Answer wapas jaata hai
    """
    # Validation
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question khali nahi ho sakta")

    if len(request.question) > 1000:
        raise HTTPException(status_code=400, detail="Question 1000 characters se zyada nahi ho sakta")

    # Initial state banao
    initial_state: RAGState = {
        "question":         request.question.strip(),
        "query_vector":     [],
        "retrieved_chunks": [],
        "rag_prompt":       "",
        "answer":           "",
    }

    try:
        # LangGraph invoke karo
        final_state = rag_graph.invoke(initial_state)
        
        return AnswerResponse(
            answer=final_state["answer"],
            chunks_retrieved=len(final_state["retrieved_chunks"]),
            question=request.question,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")