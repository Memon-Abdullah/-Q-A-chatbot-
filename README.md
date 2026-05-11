# RAG Project — Full Setup Guide

## Project Structure

```
RAG_Project/
│
├── .env                    ← Your API key goes here
├── requirements.txt        ← All Python dependencies
│
├── 1_create_vectordb.py    ← Reads DLBook.docx → chunks → embeds → saves FAISS index
├── 2_test_llm.py           ← Simple "Hello how are you" test with OpenRouter LLM
├── 3_search_vectordb.py    ← Takes user prompt → embeds → searches FAISS → prints results
├── 4_rag_pipeline.py       ← Full RAG: user prompt → search → LLM → final answer
│
└── faiss_index/            ← Auto-created by file 1 (FAISS DB saved here)
    ├── index.faiss
    └── chunks.pkl
```

---

## Step 1 — Create Virtual Environment

```bash
# Go to your project folder
cd D:\AI\Rag_Project

# Create virtual environment
python -m venv rag_env

# Activate it (Windows)
rag_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2 — Add Your API Key

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_actual_api_key_here
```

---

## Step 3 — Run Files in Order

```bash
# 1. Build the vector database from DLBook.docx
python 1_create_vectordb.py

# 2. Test LLM connection
python 2_test_llm.py

# 3. Test vector search only
python 3_search_vectordb.py

# 4. Full RAG pipeline
python 4_rag_pipeline.py
```

---

## Notes

- File 1 must be run BEFORE files 3 and 4 (it creates the FAISS index)
- The `.env` file must be present before running any file
- FAISS index is saved locally in `faiss_index/` folder — no need to rebuild unless you change the document
