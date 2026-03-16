# AI Chatbot Final Project

This project is a full-stack chatbot application built for the final submission brief.

## Features
- Chat interface with conversation history and session memory
- FastAPI backend with tool-based routing
- RAG using FAISS vector database and local knowledge files
- OCR support for uploaded images (Gemini Vision + local OCR fallback)
- PDF analysis support (PyMuPDF) for uploaded `.pdf` files
- Internet search with references
- Google-grounded search via Google Custom Search API
- Location-aware responses (location, weather, Sehri)
- Voice-to-text input from browser microphone
- Sidebar actions wired for tools/settings/help/new chat
- LangSmith tracing hooks for chains/retrieval/tools

## Tech Stack
- Frontend: HTML, CSS, JavaScript
- Backend: Python, FastAPI, Uvicorn
- LLMs: Groq (text), Gemini (vision)
- Document Parsing: PyMuPDF (`pymupdf`)
- Vector DB: FAISS
- Retrieval/Orchestration: LangChain
- Tracing: LangSmith

## Project Structure
- `Frontend/` UI and client logic
- `Backend/main.py` API routes and chatbot logic
- `Database/setup_faiss.py` vector index builder
- `Database/*.txt` RAG knowledge sources
- `Database/faiss_index/` generated vector index

## Setup

### 1) Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure environment variables
```bash
cp .env.example .env
```
Fill in these keys in `.env`:
- `GROQ_API_KEY`
- `GOOGLE_API_KEY` (for vision)
- `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_CSE_ID` (for Google grounded search)
- `LANGSMITH_API_KEY` and project/tracing variables

### 4) Build FAISS index
```bash
python Database/setup_faiss.py
```

### 5) Start backend
```bash
uvicorn Backend.main:app --host 127.0.0.1 --port 8000
```

### 6) Open frontend
Open `Frontend/index.html` in your browser (or use Live Server).

## Supported Upload Types
- Images (`image/png`, `image/jpeg`, `image/webp`) for OCR/image analysis
- PDF (`application/pdf`) for document analysis and Q&A

## LangSmith Tracing
Tracing is enabled through environment variables and `@traceable` instrumentation on:
- RAG retrieval
- Internet/Google grounded search tools
- Text chat chains
- Image chat chain
- PDF chat chain

## Brief Compliance Mapping
- Front-end chatbot interface: Implemented
- Back-end with traceability: Implemented with LangSmith hooks
- Vector database (FAISS): Implemented
- RAG: Implemented
- OCR: Implemented
- PDF analysis: Implemented (PyMuPDF)
- Internet search tool: Implemented
- Grounding with Google search: Implemented via Google CSE

## Submission Checklist
See `SUBMISSION.md` for final handoff items:
- GitHub repository link
- YouTube presentation link
- LangSmith trace links
