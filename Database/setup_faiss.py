import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# We'll create a dummy text file to act as the knowledge base for our RAG tool
DUMMY_TEXT_FILE = "Database/dummy_knowledge.txt"

def create_dummy_knowledge_base():
    with open(DUMMY_TEXT_FILE, "w") as f:
        f.write("""
        Project AI Chatbot Details:
        Deadline: March 13th at 11:59PM.
        Requirements: Front-end (UI/UX), Back-end (FastAPI, Python), Database (FAISS).
        Tools Needed: Internet Search, OCR, and RAG.
        Tracing: Must implement LangSmith tracing.
        Developer Name: Tuhin.
        """)
    return DUMMY_TEXT_FILE

def setup_faiss():
    """Reads a text file, embeds it into a FAISS index and saves to disk."""
    print("Creating dummy knowledge base text file...")
    file_path = create_dummy_knowledge_base()

    print(f"Loading documents from {file_path}...")
    loader = TextLoader(file_path)
    docs = loader.load()

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
