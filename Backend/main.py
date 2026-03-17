import logging
import os
import sqlite3
import base64
import binascii
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Imports (lazy — loaded inside functions to avoid startup crashes)
from langchain_core.messages import HumanMessage

try:
    from langsmith import traceable
except Exception:
    def traceable(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
CHAT_DB_PATH = BASE_DIR / "Database" / "chat_history.db"

# ---------------------------------------------------------------------------
# Domain allow/block lists for web search quality control
# ---------------------------------------------------------------------------
NEWS_DOMAINS = [
    "thedailystar.net", "bdnews24.com", "dhakatribune.com", "newagebd.net",
    "tbsnews.net", "thefinancialexpress.com.bd", "prothomalo.com",
    "aljazeera.com", "bbc.com", "reuters.com", "apnews.com",
    "theguardian.com", "cnn.com", "nytimes.com", "washingtonpost.com",
    "theindependent.com", "ndtv.com", "thehindu.com", "parliament.gov.bd",
    "wikipedia.org", "britannica.com",
]

BLOCKED_DOMAINS = [
    "netflix.com", "youtube.com", "amazon.com", "primevideo.com",
    "apple.com", "hulu.com", "disneyplus.com", "imdb.com",
    "rottentomatoes.com", "people.com", "tvguide.com", "tv.com",
    "spotify.com", "tiktok.com", "instagram.com", "facebook.com",
    "twitter.com", "x.com", "reddit.com", "quora.com",
    "pinterest.com", "tumblr.com", "mypikpak.com",
]

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


@traceable(name="rag_retrieval", run_type="retriever")
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
        for keyword in [
            "search", "web", "internet", "latest", "news", "find", "look up", "google",
            "weather", "forecast", "temperature", "sehri", "suhoor", "iftar", "prayer time", "near me",
            "what happened", "happened in", "happened on", "recent", "today in", "update on",
            "event", "protest", "election", "government", "politics", "crisis", "disaster",
            "current", "right now", "this week", "this month",
        ]
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


def extract_location_hint(message: str) -> str:
    normalized = re.sub(r"\s+", " ", message.strip())
    patterns = [
        r"according to\s+(.+?)\s+what(?:'s| is)\b",
        r"(?:weather|forecast|temperature|sehri|suhoor|iftar|prayer time).*?\b(?:in|on|for|at)\s+(.+?)(?:[?.!,]|$)",
        r"\b(?:in|on|for|at)\s+(.+?)(?:[?.!,]|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            location = match.group(1).strip(" .,!?")
            if location:
                return location

    return ""


def build_search_queries(message: str) -> list[str]:
    cleaned_query = extract_search_query(message)
    lowered_query = cleaned_query.lower()
    location_hint = extract_location_hint(cleaned_query)

    queries = [cleaned_query]

    if "what happened" in lowered_query or "happened on" in lowered_query or "happened in" in lowered_query:
        queries.extend(
            [
                f"{cleaned_query} news",
                f"{cleaned_query} latest news",
                f"{cleaned_query} Reuters BBC Al Jazeera",
            ]
        )

    if "bangladesh" in lowered_query:
        queries.extend(
            [
                f"{cleaned_query} site:thedailystar.net OR site:bdnews24.com OR site:dhakatribune.com",
                f"{cleaned_query} Bangladesh news",
            ]
        )

    if "bangladesh" in lowered_query and ("july 24" in lowered_query or "24 july" in lowered_query):
        queries.extend(
            [
                "July 24 Bangladesh protests news",
                "July 24 Bangladesh student protest news",
                "24 July Bangladesh The Daily Star bdnews24 Dhaka Tribune",
            ]
        )

    if location_hint:
        if any(keyword in lowered_query for keyword in ["weather", "forecast", "temperature"]):
            queries.extend(
                [
                    f"weather today in {location_hint}",
                    f"today weather forecast {location_hint}",
                ]
            )

        if any(keyword in lowered_query for keyword in ["sehri", "suhoor"]):
            queries.extend(
                [
                    f"sehri time today in {location_hint}",
                    f"suhoor time today in {location_hint}",
                    f"ramadan timetable {location_hint}",
                ]
            )

        if "iftar" in lowered_query:
            queries.extend(
                [
                    f"iftar time today in {location_hint}",
                    f"ramadan timetable {location_hint}",
                ]
            )

        if "prayer time" in lowered_query:
            queries.extend(
                [
                    f"prayer times today in {location_hint}",
                    f"namaz timetable {location_hint}",
                ]
            )

    deduped_queries: list[str] = []
    seen = set()
    for query in queries:
        normalized_query = query.strip()
        if normalized_query and normalized_query.lower() not in seen:
            deduped_queries.append(normalized_query)
            seen.add(normalized_query.lower())

    return deduped_queries


def is_ocr_request(message: str) -> bool:
    normalized_message = message.lower()
    return any(
        keyword in normalized_message
        for keyword in ["ocr", "extract text", "read text", "scan text", "text in this image"]
    )


def is_location_request(message: str) -> bool:
    normalized_message = message.lower()
    return any(
        keyword in normalized_message
        for keyword in [
            "my current location",
            "where am i",
            "where i am",
            "my location",
            "current location",
            "trace my location",
            "detect my location",
        ]
    )


def is_direct_location_request(message: str) -> bool:
    normalized_message = re.sub(r"\s+", " ", message.strip().lower())
    direct_patterns = [
        r"^what is my current location\??$",
        r"^what is my location\??$",
        r"^where am i\??$",
        r"^trace my location\??$",
        r"^detect my location\??$",
    ]
    return any(re.match(pattern, normalized_message) for pattern in direct_patterns)


def is_weather_request(message: str) -> bool:
    normalized_message = message.lower()
    return any(keyword in normalized_message for keyword in ["weather", "forecast", "temperature"])


def is_sehri_request(message: str) -> bool:
    normalized_message = message.lower()
    return any(keyword in normalized_message for keyword in ["sehri", "suhoor", "imsak"])


def apply_location_to_message(message: str, location_primary: Optional[str]) -> str:
    if not location_primary:
        return message

    updated_message = message
    replacements = [
        "my current location",
        "my location",
        "current location",
        "on my location",
        "in my location",
        "near me",
    ]

    for phrase in replacements:
        updated_message = re.sub(phrase, location_primary, updated_message, flags=re.IGNORECASE)

    return updated_message


def build_context_prompt(
    message: str,
    rag_context: str,
    web_context: str,
    has_image: bool,
    conversation_history: str = "",
    location_context: str = "",
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

    if location_context:
        sections.append(f"User location context:\n{location_context}")

    if rag_context and rag_context != "Vector database is not initialized.":
        sections.append(f"RAG context:\n{rag_context}")

    if web_context:
        sections.append(f"Web search context:\n{web_context}")

    sections.append(f"User request:\n{message}")
    return "\n\n".join(sections)


def build_pdf_prompt(message: str, pdf_text: str, conversation_history: str = "", rag_context: str = "") -> str:
    instructions = [
        "You are a helpful AI assistant.",
        "The user uploaded a PDF. Use the extracted PDF text as the primary source.",
        "If the PDF text is incomplete, say so briefly.",
        "Answer clearly and directly.",
    ]

    sections = ["\n".join(instructions)]

    if conversation_history:
        sections.append(f"Recent conversation context:\n{conversation_history}")

    if rag_context and rag_context != "Vector database is not initialized.":
        sections.append(f"RAG context:\n{rag_context}")

    sections.append(f"Extracted PDF text:\n{pdf_text[:20000]}")
    sections.append(f"User request:\n{message}")
    return "\n\n".join(sections)


def fetch_json(url: str) -> dict:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AIChatBot/1.0",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def geocode_location_name(location_name: str) -> Optional[tuple[float, float]]:
    if not location_name.strip():
        return None

    query = urlencode(
        {
            "q": location_name,
            "format": "jsonv2",
            "limit": 1,
        }
    )

    request = Request(
        f"https://nominatim.openstreetmap.org/search?{query}",
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AIChatBot/1.0",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if not payload:
                return None
            lat = float(payload[0].get("lat"))
            lon = float(payload[0].get("lon"))
            return lat, lon
    except Exception as exc:
        logger.warning("Location geocoding failed for '%s': %s", location_name, exc)
        return None


def weather_code_to_text(weather_code: Optional[int]) -> str:
    code_map = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return code_map.get(weather_code or -1, "Unavailable")


@traceable(name="google_grounded_search", run_type="tool")
def search_google_with_sources(query: str, max_results: int = 5) -> list[dict[str, str]]:
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY", "").strip()
    cse_id = os.getenv("GOOGLE_SEARCH_CSE_ID", "").strip()
    if not api_key or not cse_id:
        return []

    params = urlencode(
        {
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": max(1, min(max_results, 10)),
        }
    )

    request = Request(
        f"https://www.googleapis.com/customsearch/v1?{params}",
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AIChatBot/1.0",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("Google grounded search failed: %s", exc)
        return []

    items = payload.get("items") or []
    results: list[dict[str, str]] = []
    for item in items:
        url = (item.get("link") or "").strip()
        if not url:
            continue
        results.append(
            {
                "title": (item.get("title") or "Untitled").strip(),
                "url": url,
                "body": (item.get("snippet") or "").strip(),
            }
        )

    return results


def get_weather_response(chat: "ChatMessage") -> Optional[str]:
    if not (chat.user_location_primary or (chat.user_latitude is not None and chat.user_longitude is not None)):
        return None

    try:
        if chat.user_latitude is not None and chat.user_longitude is not None:
            location_query = f"{chat.user_latitude:.4f},{chat.user_longitude:.4f}"
        else:
            location_query = chat.user_location_primary or ""

        weather_data = fetch_json(f"https://wttr.in/{quote(location_query)}?format=j1")
        current = (weather_data.get("current_condition") or [{}])[0]
        today = (weather_data.get("weather") or [{}])[0]
        hourly = (today.get("hourly") or [{}])[0]

        area_names = [
            area.get("value", "")
            for area in ((weather_data.get("nearest_area") or [{}])[0].get("areaName") or [])
            if area.get("value")
        ]
        resolved_location = area_names[0] if area_names else (chat.user_location_primary or location_query)
        condition = ((current.get("weatherDesc") or [{"value": ""}])[0].get("value") or "Unavailable").strip()
        temp_c = current.get("temp_C", "N/A")
        feels_like_c = current.get("FeelsLikeC", "N/A")
        humidity = current.get("humidity", "N/A")
        wind_kmph = current.get("windspeedKmph", "N/A")
        rain_chance = hourly.get("chanceofrain", "N/A")

        has_primary_data = (
            condition and condition.lower() != "unavailable"
            and temp_c != "N/A"
            and feels_like_c != "N/A"
            and humidity != "N/A"
            and wind_kmph != "N/A"
        )

        if has_primary_data:
            return (
                f"Today's weather update for {resolved_location}: {condition}. "
                f"Temperature {temp_c}°C, feels like {feels_like_c}°C, humidity {humidity}%, "
                f"wind {wind_kmph} km/h, chance of rain {rain_chance}%."
            )
    except Exception as exc:
        logger.warning("Weather lookup failed from wttr.in: %s", exc)

    try:
        latitude = chat.user_latitude
        longitude = chat.user_longitude

        if latitude is None or longitude is None:
            if not chat.user_location_primary:
                return None
            resolved_coords = geocode_location_name(chat.user_location_primary)
            if not resolved_coords:
                return None
            latitude, longitude = resolved_coords

        meteo_query = urlencode(
            {
                "latitude": f"{latitude:.5f}",
                "longitude": f"{longitude:.5f}",
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                "daily": "precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 1,
            }
        )
        meteo_data = fetch_json(f"https://api.open-meteo.com/v1/forecast?{meteo_query}")

        current = meteo_data.get("current") or {}
        daily = meteo_data.get("daily") or {}

        temp_c = current.get("temperature_2m")
        feels_like_c = current.get("apparent_temperature")
        humidity = current.get("relative_humidity_2m")
        wind_kmph = current.get("wind_speed_10m")
        weather_code = current.get("weather_code")
        rain_probability = (daily.get("precipitation_probability_max") or ["N/A"])[0]

        if temp_c is None and feels_like_c is None and humidity is None and wind_kmph is None:
            return None

        resolved_location = chat.user_location_primary or f"{latitude:.4f}, {longitude:.4f}"
        condition = weather_code_to_text(weather_code)

        return (
            f"Today's weather update for {resolved_location}: {condition}. "
            f"Temperature {temp_c if temp_c is not None else 'N/A'}°C, "
            f"feels like {feels_like_c if feels_like_c is not None else 'N/A'}°C, "
            f"humidity {humidity if humidity is not None else 'N/A'}%, "
            f"wind {wind_kmph if wind_kmph is not None else 'N/A'} km/h, "
            f"chance of rain {rain_probability}%.")
    except Exception as exc:
        logger.warning("Weather lookup failed from Open-Meteo fallback: %s", exc)
        return None


def get_sehri_response(chat: "ChatMessage") -> Optional[str]:
    if not (chat.user_location_primary or (chat.user_latitude is not None and chat.user_longitude is not None)):
        return None

    try:
        if chat.user_latitude is not None and chat.user_longitude is not None:
            url = (
                "https://api.aladhan.com/v1/timings?"
                f"latitude={chat.user_latitude}&longitude={chat.user_longitude}&method=1"
            )
        else:
            city = (chat.user_location_primary or "").split(",", 1)[0].strip()
            country = ""
            if chat.user_location_primary and "," in chat.user_location_primary:
                country = chat.user_location_primary.split(",", 1)[1].strip()
            url = (
                "https://api.aladhan.com/v1/timingsByCity?"
                f"city={quote(city)}&country={quote(country)}&method=1"
            )

        prayer_data = fetch_json(url)
        timings = (prayer_data.get("data") or {}).get("timings") or {}
        imsak = timings.get("Imsak")
        fajr = timings.get("Fajr")
        readable_date = ((prayer_data.get("data") or {}).get("date") or {}).get("readable") or datetime.now().strftime("%d %b %Y")
        location_name = chat.user_location_primary or "your location"

        if not imsak and not fajr:
            return None

        parts = [f"Today's Sehri time for {location_name} on {readable_date}:"]
        if imsak:
            parts.append(f"Imsak {imsak}")
        if fajr:
            parts.append(f"Fajr {fajr}")

        return ". ".join(parts) + "."
    except Exception as exc:
        logger.warning("Sehri lookup failed: %s", exc)
        return None


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


# ---------------------------------------------------------------------------
# Web-scraping helpers for rich search results
# ---------------------------------------------------------------------------

def is_blocked(url: str) -> bool:
    """Return True if the URL belongs to a blocked entertainment/social domain."""
    url_lower = url.lower()
    return any(domain in url_lower for domain in BLOCKED_DOMAINS)


def is_news(url: str) -> bool:
    """Return True if the URL belongs to a known news/reference domain."""
    url_lower = url.lower()
    return any(domain in url_lower for domain in NEWS_DOMAINS)


def scrape_page_content(url: str, max_chars: int = 2000) -> str:
    """Fetch and extract clean body text from a URL using httpx + BeautifulSoup."""
    try:
        import httpx
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = httpx.get(url, headers=headers, timeout=8, follow_redirects=True)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove boilerplate tags
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "noscript", "iframe"]):
            tag.decompose()
        # Prefer <article> or <main>, fall back to <body>
        container = soup.find("article") or soup.find("main") or soup.body
        text = " ".join((container or soup).get_text(separator=" ").split())
        return text[:max_chars]
    except Exception as exc:
        logger.debug("scrape_page_content failed for %s: %s", url, exc)
        return ""


def is_news_query(query: str) -> bool:
    lowered = query.lower()
    return any(
        key in lowered
        for key in [
            "news", "latest", "what happened", "happened on", "happened in",
            "recent", "update", "protest", "election", "government", "politics",
            "crisis", "war", "flood", "earthquake", "accident",
        ]
    )


def search_news_with_sources(query: str, max_results: int = 5) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    try:
        from ddgs import DDGS

        for item in DDGS().news(query, max_results=max_results * 2):
            url = (item.get("url") or item.get("href") or "").strip()
            if not url or is_blocked(url):
                continue
            title = (item.get("title") or "Untitled").strip()
            body = (item.get("body") or "").strip()
            source = (item.get("source") or "").strip()
            if source and source.lower() not in title.lower():
                title = f"{title} ({source})"
            results.append({"title": title, "href": url, "body": body})
            if len(results) >= max_results:
                break
    except Exception as exc:
        logger.warning("DDGS news search failed for '%s': %s", query, exc)
    return results


@traceable(name="internet_search_tool", run_type="tool")
def web_search(query: str) -> str:
    try:
        from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun().invoke(query) 
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        return ""


@traceable(name="search_with_references", run_type="tool")
def search_web_with_sources(query: str, max_results: int = 5) -> tuple[str, list[dict[str, str]]]:
    cleaned_query = extract_search_query(query)
    try:
        results = []

        search_queries = build_search_queries(query)
        if is_news_query(cleaned_query):
            for query_variant in search_queries[:8]:
                results.extend(search_news_with_sources(query_variant, max_results=max_results))
        lowered_query = cleaned_query.lower()
        if "bangladesh" in lowered_query and "parliament" in lowered_query and ("member" in lowered_query or "name" in lowered_query):
            search_queries.extend([
                "site:parliament.gov.bd member list",
                "site:wikipedia.org list of members of the jatiya sangsad",
                "bangladesh parliament members list official",
            ])

        google_seen_urls = set()
        for query_variant in search_queries:
            for item in search_google_with_sources(query_variant, max_results=max_results):
                url = (item.get("url") or "").strip()
                if not url or url in google_seen_urls:
                    continue
                google_seen_urls.add(url)
                results.append(
                    {
                        "title": item.get("title", "Untitled"),
                        "href": url,
                        "body": item.get("body", ""),
                    }
                )

        try:
            from ddgs import DDGS

            for query_variant in search_queries:
                for backend in ["bing", "duckduckgo", "brave", "yahoo"]:
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
            # Skip entertainment/social media domains
            if is_blocked(url):
                continue
            seen_urls.add(url)

            searchable_text = f"{title} {snippet} {url}".lower()
            if "bangladesh" in cleaned_query.lower() and "bangladesh" not in searchable_text:
                if not any(domain in url.lower() for domain in ["thedailystar.net", "bdnews24.com", "dhakatribune.com", "newagebd.net", "tbsnews.net", "thefinancialexpress.com.bd", "prothomalo.com", "parliament.gov.bd"]):
                    continue
            if any(key in cleaned_query.lower() for key in ["what happened", "happened on", "happened in"]):
                if not (is_news(url) or any(word in searchable_text for word in ["news", "protest", "government", "student", "bangladesh", "police", "court", "election"])):
                    continue
            if "bangladesh" in cleaned_query.lower() and any(key in cleaned_query.lower() for key in ["july 24", "24 july"]):
                if not any(word in searchable_text for word in ["july 24", "24 july", "protest", "student", "curfew", "hasina", "unrest", "violations", "revolution"]):
                    continue
            overlap = sum(1 for token in query_tokens if token in searchable_text)
            domain_bonus = 1 if any(domain in url.lower() for domain in ["wikipedia.org", ".gov", "parliament.gov.bd"]) else 0
            list_bonus = 1 if any(key in searchable_text for key in ["list of members", "member list", "jatiya sangsad"]) else 0
            news_bonus = 2 if is_news(url) else 0
            country_bonus = 2 if "bangladesh" in cleaned_query.lower() and "bangladesh" in searchable_text else 0
            date_bonus = 2 if any(key in cleaned_query.lower() for key in ["july 24", "24 july"]) and any(key in searchable_text for key in ["july 24", "24 july"]) else 0
            penalty = 0
            if any(bad in searchable_text for bad in ["grammar", "tense", "english", "idiom", "tutor", "learn english"]):
                penalty += 4
            if not any(word in cleaned_query.lower() for word in ["sport", "cricket", "football", "match", "game"]):
                if any(bad in searchable_text for bad in ["cricket", "asia cup", "vs sri lanka", "match preview", "sports"]):
                    penalty += 4
            score = overlap + domain_bonus + list_bonus + news_bonus + country_bonus + date_bonus - penalty
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

    lines = ["**References:**"]
    for index, item in enumerate(sources, start=1):
        title = item.get("title", "Source")
        url = item.get("url", "")
        lines.append(f"{index}. [{title}]({url})")
    return "\n".join(lines)


@traceable(name="full_web_search_chain", run_type="chain")
def run_full_web_search_chain(message: str, location_context: str = "") -> str:
    """
    Full pipeline:
      1. Search for relevant URLs (filtered, ranked)
      2. Scrape body text from top pages
      3. Ask Groq to summarise based on scraped content
      4. Always append numbered markdown reference links
    """
    web_context, web_sources = search_web_with_sources(message)

    # Scrape body text for the top sources
    scraped_parts: list[str] = []
    for idx, source in enumerate(web_sources[:5], start=1):
        url = source.get("url", "")
        if not url:
            continue
        page_text = scrape_page_content(url, max_chars=2000)
        title = source.get("title", "Source")
        if page_text:
            scraped_parts.append(f"[{idx}] {title}\nURL: {url}\n{page_text}")
        elif web_context:
            # Fall back to snippet already in web_context
            pass

    scraped_context = "\n\n".join(scraped_parts) if scraped_parts else web_context

    if not scraped_context and not web_sources:
        return "I could not fetch reliable web results right now. Please try again in a moment."

    date_str = datetime.now().strftime("%d %B %Y")
    location_hint = f"\nUser location: {location_context}" if location_context else ""
    prompt = (
        f"Today is {date_str}.{location_hint}\n"
        "You are a knowledgeable assistant that always cites sources.\n"
        "Below is content scraped from real news portals and web pages.\n"
        "Read it carefully and answer the user's question accurately and specifically.\n"
        "If the content mentions specific events, names, dates, or figures — include them.\n"
        "Do NOT mention TV shows, movies, or entertainment unless the user explicitly asked about them.\n\n"
        f"=== Scraped Web Content ===\n{scraped_context}\n\n"
        f"User question: {message}\n\n"
        "Write a clear, detailed answer based on the above content. "
        "Do not hallucinate — if information is not in the content, say so."
    )

    try:
        resp = get_groq_llm().invoke([HumanMessage(content=prompt)])
        answer = normalize_response_content(resp.content)
    except Exception as exc:
        logger.warning("LLM call failed in run_full_web_search_chain: %s", exc)
        answer = scraped_context[:1500] if scraped_context else "Web search returned no useful content."

    references = format_reference_links(web_sources)
    if references:
        answer = f"{answer}\n\n{references}"
    return answer


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


def validate_and_decode_pdf(document_base64: str, document_mime_type: str) -> bytes:
    allowed_types = {"application/pdf"}
    if document_mime_type not in allowed_types:
        raise RuntimeError(f"Unsupported document type: {document_mime_type}. Allowed: application/pdf")

    try:
        document_bytes = base64.b64decode(document_base64, validate=True)
    except (binascii.Error, ValueError):
        raise RuntimeError("Invalid document payload: document_base64 is not valid base64")

    max_size = 12 * 1024 * 1024
    if len(document_bytes) > max_size:
        raise RuntimeError("PDF too large. Maximum allowed size is 12MB")

    return document_bytes


def extract_pdf_text(document_bytes: bytes) -> str:
    try:
        import fitz
    except Exception as exc:
        logger.warning("PyMuPDF is unavailable for PDF extraction: %s", exc)
        raise RuntimeError("PDF analysis requires PyMuPDF. Install with: pip install pymupdf")

    try:
        text_parts: list[str] = []
        with fitz.open(stream=document_bytes, filetype="pdf") as doc:
            for page in doc:
                page_text = page.get_text("text").strip()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts).strip()
    except Exception as exc:
        logger.warning("PDF text extraction failed: %s", exc)
        raise RuntimeError("Could not parse the PDF file.")


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


@traceable(name="text_chat_chain", run_type="chain")
def do_text_chat(message: str, session_id: str) -> str:
    """Text-only path: Groq + RAG + optional web search (with page scraping + references)."""
    use_search = should_use_search(message)
    if use_search:
        return run_full_web_search_chain(message)

    rag_context = get_rag_response(message) or get_text_fallback_response()
    history = get_recent_chat_history(session_id, limit=12)
    prompt = build_context_prompt(
        message,
        rag_context=rag_context,
        web_context="",
        has_image=False,
        conversation_history=format_chat_history(history),
    )
    resp = get_groq_llm().invoke([HumanMessage(content=prompt)])
    return normalize_response_content(resp.content)


@traceable(name="image_chat_chain", run_type="chain")
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


@traceable(name="pdf_chat_chain", run_type="chain")
def do_pdf_chat(message: str, document_base64: str, document_mime_type: str, session_id: str) -> str:
    document_bytes = validate_and_decode_pdf(document_base64, document_mime_type)
    extracted_text = extract_pdf_text(document_bytes)
    if not extracted_text:
        raise RuntimeError("No readable text found in the uploaded PDF.")

    history = get_recent_chat_history(session_id, limit=12)
    rag_context = get_rag_response(message)
    prompt = build_pdf_prompt(
        message=message,
        pdf_text=extracted_text,
        conversation_history=format_chat_history(history),
        rag_context=rag_context,
    )

    response = get_groq_llm().invoke([HumanMessage(content=prompt)])
    return normalize_response_content(response.content)


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
    user_location_primary: Optional[str] = None
    user_location_secondary: Optional[str] = None
    user_latitude: Optional[float] = None
    user_longitude: Optional[float] = None
    image_base64: Optional[str] = None
    image_mime_type: Optional[str] = None
    image_name: Optional[str] = None
    document_base64: Optional[str] = None
    document_mime_type: Optional[str] = None
    document_name: Optional[str] = None


def build_location_context(chat: ChatMessage) -> str:
    parts = []
    if chat.user_location_primary:
        parts.append(f"Primary location: {chat.user_location_primary}")
    if chat.user_location_secondary:
        parts.append(f"Location source: {chat.user_location_secondary}")
    if chat.user_latitude is not None and chat.user_longitude is not None:
        parts.append(f"Coordinates: {chat.user_latitude}, {chat.user_longitude}")
    return "\n".join(parts)


def get_location_response(chat: ChatMessage) -> Optional[str]:
    if not is_direct_location_request(chat.message):
        return None

    if chat.user_location_primary:
        response = f"Your current detected location is {chat.user_location_primary}."
        if chat.user_location_secondary:
            response += f" {chat.user_location_secondary}."
        if chat.user_latitude is not None and chat.user_longitude is not None:
            response += f" Coordinates: {chat.user_latitude:.5f}, {chat.user_longitude:.5f}."
        return response

    return (
        "I do not have your current location yet. Please click 'Update location' in the sidebar "
        "and allow browser location access, then ask again."
    )


@traceable(name="text_chat_with_context_chain", run_type="chain")
def do_text_chat_with_context(message: str, session_id: str, location_context: str = "") -> str:
    use_search = should_use_search(message)
    if use_search:
        return run_full_web_search_chain(message, location_context=location_context)

    rag_context = get_rag_response(message) or get_text_fallback_response()
    history = get_recent_chat_history(session_id, limit=12)
    prompt = build_context_prompt(
        message,
        rag_context=rag_context,
        web_context="",
        has_image=False,
        conversation_history=format_chat_history(history),
        location_context=location_context,
    )
    resp = get_groq_llm().invoke([HumanMessage(content=prompt)])
    return normalize_response_content(resp.content)


@app.post("/chat")
async def chat_endpoint(chat: ChatMessage):
    message = chat.message.strip() or "Analyze this image."
    session_id = (chat.session_id or "default-session").strip() or "default-session"
    location_context = build_location_context(chat)
    enriched_message = apply_location_to_message(message, chat.user_location_primary)

    user_content = enriched_message
    if chat.image_name:
        user_content = f"{enriched_message}\n[Image: {chat.image_name}]"
    elif chat.document_name:
        user_content = f"{enriched_message}\n[PDF: {chat.document_name}]"

    save_chat_message(session_id, "user", user_content)

    try:
        location_response = get_location_response(chat)
        if location_response:
            save_chat_message(session_id, "assistant", location_response)
            return {"response": location_response, "session_id": session_id}

        if is_sehri_request(enriched_message):
            sehri_response = get_sehri_response(chat)
            if sehri_response:
                save_chat_message(session_id, "assistant", sehri_response)
                return {"response": sehri_response, "session_id": session_id}

        if is_weather_request(enriched_message):
            weather_response = get_weather_response(chat)
            if weather_response:
                save_chat_message(session_id, "assistant", weather_response)
                return {"response": weather_response, "session_id": session_id}

        if chat.document_base64 and chat.document_mime_type:
            response_text = do_pdf_chat(enriched_message, chat.document_base64, chat.document_mime_type, session_id)
        elif chat.image_base64 and chat.image_mime_type:
            response_text = do_image_chat(enriched_message, chat.image_base64, chat.image_mime_type, session_id)
        else:
            response_text = do_text_chat_with_context(enriched_message, session_id, location_context)

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
