from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

DB_DIR = Path("Database")
DEFAULT_KNOWLEDGE_FILE = DB_DIR / "dummy_knowledge.txt"


def ensure_default_knowledge_file() -> None:
    if DEFAULT_KNOWLEDGE_FILE.exists():
        return
    DEFAULT_KNOWLEDGE_FILE.write_text(
        """
Project AI Chatbot Details:
Deadline: March 13th at 11:59PM.
Requirements: Front-end (UI/UX), Back-end (FastAPI, Python), Database (FAISS).
Tools Needed: Internet Search, OCR, and RAG.
Tracing: Must implement LangSmith tracing.
Developer Name: Tuhin.
        """.strip(),
        encoding="utf-8",
    )


def load_knowledge_documents():
    ensure_default_knowledge_file()
    docs = []
    for txt_path in sorted(DB_DIR.glob("*.txt")):
        loader = TextLoader(str(txt_path), encoding="utf-8")
        docs.extend(loader.load())
    return docs

def setup_faiss():
    """Reads knowledge text files, embeds them into a FAISS index and saves to disk."""
    print("Loading knowledge documents from Database/*.txt...")
    docs = load_knowledge_documents()
    print(f"Loaded {len(docs)} document(s).")

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    print("Embedding chunks and initializing FAISS database using HuggingFace Local Embeddings...")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)

    index_path = "Database/faiss_index"
    print(f"Saving FAISS index locally to {index_path}...")
    vectorstore.save_local(index_path)
    print("Vector database setup complete!")

if __name__ == "__main__":
    setup_faiss()
