import logging
import os
import sqlite3
import base64
import binascii
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Imports (lazy — loaded inside functions to avoid startup crashes)
from langchain_core.messages import HumanMessage

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
CHAT_DB_PATH = BASE_DIR / "Database" / "chat_history.db"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Agent Chatbot API")

FRONTEND_DIR = BASE_DIR / "Frontend"
if FRONTEND_DIR.exists():
    app.mount("/Frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


def init_chat_db() -> None:
    CHAT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_chat_message(session_id: str, role: str, content: str) -> None:
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO chat_history (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, role, content, datetime.utcnow().isoformat()),
        )
        conn.commit()


def get_recent_chat_history(session_id: str, limit: int = 12) -> list[tuple[str, str]]:
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM chat_history
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    return list(reversed(rows))


def format_chat_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return ""

    lines = []
    for role, content in history:
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


init_chat_db()

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


def extract_search_query(message: str) -> str:
    normalized = message.strip()
    lowered = normalized.lower()
    prefixes = [
        "search the web for:",
        "search web for:",
        "search for:",
    ]
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return normalized[len(prefix):].strip() or normalized
    return normalized


def is_ocr_request(message: str) -> bool:
    normalized_message = message.lower()
    return any(
        keyword in normalized_message
        for keyword in ["ocr", "extract text", "read text", "scan text", "text in this image"]
    )


def build_context_prompt(
    message: str,
    rag_context: str,
    web_context: str,
    has_image: bool,
    conversation_history: str = "",
) -> str:
    instructions = [
        "You are a helpful AI assistant for this project.",
        "Answer naturally and directly, similar to Gemini.",
        "If project context is relevant, use the retrieved RAG context.",
        "When retrieved context contains explicit values (dates, names, counts), quote them exactly.",
        "If web search context is provided, use it carefully and summarize the useful parts.",
        "If the answer is not certain, say so briefly.",
    ]

    if has_image and is_ocr_request(message):
        instructions.append("The user wants OCR. Extract the visible text accurately, then answer the user's question if needed.")
    elif has_image:
        instructions.append("Analyze the provided image and answer the user's request.")

    if web_context:
        instructions.append(
            "This is a web-search request. Use ONLY the web search context for factual claims. "
            "If data is missing, say it clearly."
        )

    sections = ["\n".join(instructions)]

    if conversation_history:
        sections.append(f"Recent conversation context:\n{conversation_history}")

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
            search_result = web_search(message)
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


def search_web_with_sources(query: str, max_results: int = 5) -> tuple[str, list[dict[str, str]]]:
    cleaned_query = extract_search_query(query)
    try:
        results = []

        search_queries = [cleaned_query]
        lowered_query = cleaned_query.lower()
        if "bangladesh" in lowered_query and "parliament" in lowered_query and ("member" in lowered_query or "name" in lowered_query):
            search_queries.extend([
                "site:parliament.gov.bd member list",
                "site:wikipedia.org list of members of the jatiya sangsad",
                "bangladesh parliament members list official",
            ])

        try:
            from ddgs import DDGS

            for query_variant in search_queries:
                for backend in ["auto", "lite", "bing"]:
                    try:
                        attempt = DDGS().text(query_variant, max_results=max_results * 2, backend=backend)
                        if attempt:
                            results.extend(attempt)
                            break
                    except Exception as backend_exc:
                        logger.warning("ddgs backend '%s' failed for '%s': %s", backend, query_variant, backend_exc)
        except Exception as exc:
            logger.warning("ddgs package unavailable: %s", exc)

        if not results:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                for query_variant in search_queries:
                    results.extend(list(ddgs.text(query_variant, max_results=max_results * 2)))

        sources: list[dict[str, str]] = []
        context_parts = []

        seen_urls = set()
        query_tokens = {token for token in re.findall(r"[a-z0-9]+", cleaned_query.lower()) if len(token) > 2}

        ranked_items = []
        for item in results:
            title = (item.get("title") or "Untitled").strip()
            url = (item.get("href") or item.get("url") or "").strip()
            snippet = (item.get("body") or "").strip()
            if not url:
                continue

            if url in seen_urls:
                continue
            seen_urls.add(url)

            searchable_text = f"{title} {snippet} {url}".lower()
            overlap = sum(1 for token in query_tokens if token in searchable_text)
            domain_bonus = 1 if any(domain in url.lower() for domain in ["wikipedia.org", ".gov", "parliament.gov.bd"]) else 0
            list_bonus = 1 if any(key in searchable_text for key in ["list of members", "member list", "jatiya sangsad"]) else 0
            score = overlap + domain_bonus + list_bonus
            ranked_items.append((score, title, url, snippet))

        ranked_items.sort(key=lambda row: row[0], reverse=True)

        if ranked_items and ranked_items[0][0] == 0:
            chosen = ranked_items[:max_results]
        else:
            chosen = [row for row in ranked_items if row[0] > 0][:max_results]
            if not chosen:
                chosen = ranked_items[:max_results]

        for index, (_score, title, url, snippet) in enumerate(chosen, start=1):

            sources.append({"title": title, "url": url})
            context_parts.append(f"[{index}] {title}\nURL: {url}\nSnippet: {snippet}")

        return "\n\n".join(context_parts), sources
    except Exception as exc:
        logger.warning("Structured web search failed: %s", exc)
        fallback = web_search(cleaned_query)
        return fallback, []


def format_reference_links(sources: list[dict[str, str]]) -> str:
    if not sources:
        return ""

    lines = ["References:"]
    for index, item in enumerate(sources, start=1):
        title = item.get("title", "Source")
        url = item.get("url", "")
        lines.append(f"{index}. {title} - {url}")
    return "\n".join(lines)


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

    vision_model = os.getenv("GOOGLE_VISION_MODEL") or os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
    return ChatGoogleGenerativeAI(
        model=vision_model,
        temperature=0,
        google_api_key=api_key,
        max_retries=0,
    )


def validate_and_decode_image(image_base64: str, image_mime_type: str) -> bytes:
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    if image_mime_type not in allowed_types:
        raise RuntimeError(f"Unsupported image type: {image_mime_type}. Allowed: png/jpeg/webp")

    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError):
        raise RuntimeError("Invalid image payload: image_base64 is not valid base64")

    max_size = 8 * 1024 * 1024
    if len(image_bytes) > max_size:
        raise RuntimeError("Image too large. Maximum allowed size is 8MB")

    return image_bytes


def local_ocr_extract_text(image_bytes: bytes) -> str:
    try:
        from PIL import Image
    except Exception as exc:
        logger.warning("Pillow is unavailable for OCR: %s", exc)
        return ""

    image = Image.open(io.BytesIO(image_bytes))

    try:
        import pytesseract

        if not os.getenv("TESSERACT_CMD"):
            for candidate in ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"]:
                if Path(candidate).exists():
                    pytesseract.pytesseract.tesseract_cmd = candidate
                    break

        text = pytesseract.image_to_string(image).strip()
        if text:
            return text
    except Exception as exc:
        logger.warning("Tesseract OCR unavailable/failed: %s", exc)

    try:
        import numpy as np
        import easyocr

        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(np.array(image), detail=0)
        return "\n".join(part.strip() for part in results if part and part.strip())
    except Exception as exc:
        logger.warning("EasyOCR fallback failed: %s", exc)
        return ""


def do_text_chat(message: str, session_id: str) -> str:
    """Text-only path: Groq + RAG + optional web search."""
    use_search = should_use_search(message)
    rag_context = "" if use_search else (get_rag_response(message) or get_text_fallback_response())
    web_context = ""
    web_sources: list[dict[str, str]] = []
    if use_search:
        web_context, web_sources = search_web_with_sources(message)
        if not web_sources and not web_context:
            return (
                "I could not fetch reliable web results right now. Please try again in a moment."
            )

    history = get_recent_chat_history(session_id, limit=12)
    prompt = build_context_prompt(
        message,
        rag_context=rag_context,
        web_context=web_context,
        has_image=False,
        conversation_history=format_chat_history(history),
    )
    resp = get_groq_llm().invoke([HumanMessage(content=prompt)])
    answer = normalize_response_content(resp.content)
    references = format_reference_links(web_sources)
    if references:
        answer = f"{answer}\n\n{references}"
    return answer


def do_image_chat(message: str, image_base64: str, image_mime_type: str, session_id: str) -> str:
    """Image path: Gemini Vision first, then local OCR fallback when Gemini fails."""
    image_bytes = validate_and_decode_image(image_base64, image_mime_type)
    rag_context = get_rag_response(message)
    history = get_recent_chat_history(session_id, limit=8)
    prompt = build_context_prompt(
        message,
        rag_context=rag_context,
        web_context="",
        has_image=True,
        conversation_history=format_chat_history(history),
    )
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": f"data:{image_mime_type};base64,{image_base64}"},
    ]

    try:
        resp = get_gemini_llm().invoke([HumanMessage(content=content)])
        return normalize_response_content(resp.content)
    except Exception as exc:
        logger.warning("Gemini vision failed, trying local OCR fallback: %s", exc)

    extracted_text = local_ocr_extract_text(image_bytes)
    if not extracted_text:
        raise RuntimeError(
            "Image analysis failed: Gemini Vision unavailable and local OCR returned no readable text"
        )

    fallback_prompt = (
        "You are helping with image analysis. "
        "Use the OCR text below and answer the user's request clearly.\n\n"
        f"OCR text:\n{extracted_text}\n\n"
        f"User request:\n{message}"
    )
    fallback_response = get_groq_llm().invoke([HumanMessage(content=fallback_prompt)])
    return normalize_response_content(fallback_response.content)


def build_error_payload(code: str, message: str, details: Optional[str] = None) -> dict:
    payload = {
        "error": message,
        "error_code": code,
    }
    if details:
        payload["error_details"] = details
    return payload

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    message: str = ""
    session_id: Optional[str] = None
    image_base64: Optional[str] = None
    image_mime_type: Optional[str] = None
    image_name: Optional[str] = None


@app.post("/chat")
async def chat_endpoint(chat: ChatMessage):
    message = chat.message.strip() or "Analyze this image."
    session_id = (chat.session_id or "default-session").strip() or "default-session"

    user_content = message
    if chat.image_name:
        user_content = f"{message}\n[Image: {chat.image_name}]"

    save_chat_message(session_id, "user", user_content)

    try:
        if chat.image_base64 and chat.image_mime_type:
            response_text = do_image_chat(message, chat.image_base64, chat.image_mime_type, session_id)
        else:
            response_text = do_text_chat(message, session_id)

        save_chat_message(session_id, "assistant", response_text)
        return {"response": response_text, "session_id": session_id}
    except RuntimeError as e:
        return build_error_payload("runtime_error", str(e))
    except Exception as e:
        err = str(e)
        logger.exception("Chat request failed")
        is_quota = any(x in err for x in ["429", "quota", "RESOURCE_EXHAUSTED"])
        if is_quota and chat.image_base64:
            return build_error_payload(
                "vision_quota_exhausted",
                "Gemini Vision quota exhausted. Add a paid-tier GOOGLE_API_KEY for image/OCR.",
            )
        if is_quota:
            rag = get_rag_response(message) or get_text_fallback_response()
            fallback_text = "AI rate-limited. Local knowledge:\n\n" + rag
            save_chat_message(session_id, "assistant", fallback_text)
            return {"response": fallback_text, "session_id": session_id}
        return build_error_payload("chat_failed", "Chat request failed", err[:300])


@app.get("/chat/history/{session_id}")
def get_chat_history(session_id: str):
    rows = get_recent_chat_history(session_id=session_id, limit=100)
    return {
        "session_id": session_id,
        "messages": [{"role": role, "content": content} for role, content in rows],
    }


@app.get("/")
def root():
    if FRONTEND_DIR.exists():
        return RedirectResponse(url="/Frontend/index.html")
    return {
        "status": "running",
        "groq_ready": bool(os.getenv("GROQ_API_KEY")),
        "gemini_ready": bool(os.getenv("GOOGLE_API_KEY")),
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
