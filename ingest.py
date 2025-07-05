# ingestion.py
import os
import fitz  # PyMuPDF
import json
import logging
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import EmbeddingFunction

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [INGEST] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load embedding model
logger.info("Loading sentence transformer model...")
local_model = SentenceTransformer("all-MiniLM-L6-v2")

class LocalEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts):
        logger.info("Generating local embeddings...")
        if isinstance(texts, str):
            texts = [texts]
        return local_model.encode(texts).tolist()

# Instantiate embedding function
local_ef = LocalEmbeddingFunction()

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=local_ef)

# Load PDF or JSON documents from directory
def load_documents_from_directory(directory_path):
    logger.info("Loading documents from directory...")
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.endswith(".pdf"):
            logger.info(f"Reading PDF: {filename}")
            try:
                with fitz.open(file_path) as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    documents.append({"id": filename, "text": text})
            except Exception as e:
                logger.error(f"Failed to read PDF {filename}: {e}")
        elif filename.endswith(".json"):
            logger.info(f"Reading JSON: {filename}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                    # 👇 Extract relevant text fields - customize as needed
                    if isinstance(json_data, dict):
                        text = json.dumps(json_data)
                    else:
                        text = str(json_data)
                    documents.append({"id": filename, "text": text})
            except Exception as e:
                logger.error(f"Failed to read JSON {filename}: {e}")
        else:
            logger.warning(f"Skipping unsupported file type: {filename}")
    return documents

# Split large text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Ingest documents into ChromaDB
def ingest_documents(directory_path="./documents"):
    documents = load_documents_from_directory(directory_path)
    logger.info(f"Total documents loaded: {len(documents)}")

    chunked_documents = []
    for doc in documents:
        logger.info(f"Splitting {doc['id']} into chunks...")
        chunks = split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunked_documents.append({
                "id": f"{doc['id']}_chunk{i+1}",
                "text": chunk
            })

    for doc in chunked_documents:
        doc["embedding"] = local_ef(doc["text"])

    logger.info("Upserting documents into ChromaDB...")
    for doc in chunked_documents:
        collection.upsert(
            ids=[doc["id"]],
            documents=[doc["text"]],
            embeddings=doc["embedding"]
        )
    logger.info("Ingestion complete.")

# Run ingestion if file is executed directly
if __name__ == "__main__":
    ingest_documents()
