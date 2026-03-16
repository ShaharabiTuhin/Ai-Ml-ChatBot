import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Imports (lazy — loaded inside functions to avoid startup crashes)
from langchain_core.messages import HumanMessage

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Agent Chatbot API")

# Setup CORS so the frontend can talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ---------------------------------------------------------------------------
# AI Agent Tools
# ---------------------------------------------------------------------------

def get_retriever():
    """Load the FAISS retriever lazily so the API can still start if embeddings are unavailable."""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        index_path = BASE_DIR / "Database" / "faiss_index"
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore.as_retriever(search_kwargs={"k": 2})
    except Exception as exc:
        logger.warning("FAISS retriever unavailable: %s", exc)
        return None


def get_rag_response(query: str) -> str:
    retriever = get_retriever()
    if not retriever:
        return "Vector database is not initialized."

    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])


def get_text_fallback_response() -> str:
    knowledge_file = BASE_DIR / "Database" / "dummy_knowledge.txt"
    if knowledge_file.exists():
        return knowledge_file.read_text(encoding="utf-8").strip()
    return "Local knowledge base is unavailable."


def normalize_response_content(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join([part for part in parts if part]).strip()

    return str(content)


def should_use_search(message: str) -> bool:
    normalized_message = message.lower()
    return any(
        keyword in normalized_message
        for keyword in ["search", "web", "internet", "latest", "news", "find", "look up", "google"]
    )


def is_ocr_request(message: str) -> bool:
    normalized_message = message.lower()
    return any(
        keyword in normalized_message
        for keyword in ["ocr", "extract text", "read text", "scan text", "text in this image"]
    )


def build_context_prompt(message: str, rag_context: str, web_context: str, has_image: bool) -> str:
    instructions = [
        "You are a helpful AI assistant for this project.",
        "Answer naturally and directly, similar to Gemini.",
        "If project context is relevant, use the retrieved RAG context.",
        "If web search context is provided, use it carefully and summarize the useful parts.",
        "If the answer is not certain, say so briefly.",
    ]

    if has_image and is_ocr_request(message):
        instructions.append("The user wants OCR. Extract the visible text accurately, then answer the user's question if needed.")
    elif has_image:
        instructions.append("Analyze the provided image and answer the user's request.")

    sections = ["\n".join(instructions)]

    if rag_context and rag_context != "Vector database is not initialized.":
        sections.append(f"RAG context:\n{rag_context}")

    if web_context:
        sections.append(f"Web search context:\n{web_context}")

    sections.append(f"User request:\n{message}")
    return "\n\n".join(sections)


def get_local_fallback_response(message: str) -> str:
    normalized_message = message.strip().lower()
    now = datetime.now()

    if any(keyword in normalized_message for keyword in ["time", "current time", "what time"]):
        return f"The current local time is {now.strftime('%I:%M %p')} on {now.strftime('%A, %d %B %Y')}."

    if any(keyword in normalized_message for keyword in ["date", "today", "day", "what day"]):
        return f"Today's local date is {now.strftime('%A, %d %B %Y')}."

    if any(keyword in normalized_message for keyword in ["deadline", "due", "submission"]):
        return "The project deadline in the local knowledge base is March 13th at 11:59 PM."

    if any(keyword in normalized_message for keyword in ["developer", "who made", "who built", "author"]):
        return "The developer name in the project knowledge base is Tuhin."

    if any(keyword in normalized_message for keyword in ["requirement", "requirements", "stack", "tech stack"]):
        return "The listed requirements are Front-end (UI/UX), Back-end (FastAPI, Python), and Database (FAISS)."

    if any(keyword in normalized_message for keyword in ["trace", "langsmith"]):
        return "LangSmith tracing is required for this project and is enabled through the environment configuration."

    if any(keyword in normalized_message for keyword in ["ocr", "image", "scan", "extract text"]):
        return "OCR is configured as a placeholder tool right now. It can be upgraded with a real image-to-text library if you add sample files."

    if any(keyword in normalized_message for keyword in ["search", "web", "internet", "news", "google"]):
        try:
            search_result = search_tool.invoke(message)
            if search_result:
                return f"Web search result:\n\n{search_result}"
        except Exception as exc:
            logger.warning("Local web search fallback failed: %s", exc)

    rag_response = get_rag_response(message)
    if rag_response != "Vector database is not initialized.":
        return f"Local knowledge base answer:\n\n{rag_response}"

    return (
        "The AI model is unavailable right now, but the backend is running. "
        "Try asking about the project deadline, developer, requirements, tracing, OCR, date, time, or web search."
    )


def web_search(query: str) -> str:
    try:
        from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun().invoke(query)
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        return ""


def get_groq_llm():
    """Primary LLM for all text chat — Groq free tier."""
    from langchain_groq import ChatGroq
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Get a free key at https://console.groq.com")
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
        groq_api_key=api_key,
    )


def get_gemini_llm():
    """Vision LLM — used only for image analysis / OCR."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        temperature=0,
        google_api_key=api_key,
        max_retries=0,
    )


def do_text_chat(message: str) -> str:
    """Text-only path: Groq + RAG + optional web search."""
    rag_context = get_rag_response(message) or get_text_fallback_response()
    web_context = web_search(message) if should_use_search(message) else ""
    prompt = build_context_prompt(message, rag_context=rag_context, web_context=web_context, has_image=False)
    resp = get_groq_llm().invoke([HumanMessage(content=prompt)])
    return normalize_response_content(resp.content)


def do_image_chat(message: str, image_base64: str, image_mime_type: str) -> str:
    """Image path: Gemini Vision only."""
    rag_context = get_rag_response(message)
    prompt = build_context_prompt(message, rag_context=rag_context, web_context="", has_image=True)
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": f"data:{image_mime_type};base64,{image_base64}"},
    ]
    resp = get_gemini_llm().invoke([HumanMessage(content=content)])
    return normalize_response_content(resp.content)

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    message: str = ""
    image_base64: Optional[str] = None
    image_mime_type: Optional[str] = None
    image_name: Optional[str] = None


@app.post("/chat")
async def chat_endpoint(chat: ChatMessage):
    message = chat.message.strip() or "Analyze this image."
    try:
        if chat.image_base64 and chat.image_mime_type:
            return {"response": do_image_chat(message, chat.image_base64, chat.image_mime_type)}
        return {"response": do_text_chat(message)}
    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        err = str(e)
        logger.exception("Chat request failed")
        is_quota = any(x in err for x in ["429", "quota", "RESOURCE_EXHAUSTED"])
        if is_quota and chat.image_base64:
            return {"error": "Gemini Vision quota exhausted. Add a paid-tier GOOGLE_API_KEY for image/OCR."}
        if is_quota:
            rag = get_rag_response(message) or get_text_fallback_response()
            return {"response": "AI rate-limited. Local knowledge:\n\n" + rag}
        return {"error": f"Error: {err[:300]}"}


@app.get("/")
def root():
    return {
        "status": "running",
        "groq_ready": bool(os.getenv("GROQ_API_KEY")),
        "gemini_ready": bool(os.getenv("GOOGLE_API_KEY")),
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
