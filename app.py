import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from groq import Groq
from chromadb.utils.embedding_functions import EmbeddingFunction
import fitz
import logging
from langdetect import detect
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
from datetime import datetime



load_dotenv()


#multi-lingual support

# Constants
MODEL_NAME = "facebook/nllb-200-distilled-600M"
TARGET_LANG_CODE = "eng_Latn"

# Load model and tokenizer globally
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Language mapping: ISO 639-1 -> NLLB code
LANG_CODE_MAP = {
    'en': 'eng_Latn',
    'hi': 'hin_Deva',
    'ta': 'tam_Taml',
    'bn': 'ben_Beng',
    'te': 'tel_Telu',
    'kn': 'kan_Knda',
    'ml': 'mal_Mlym',
    'gu': 'guj_Gujr',
    'mr': 'mar_Deva',
    'ur': 'urd_Arab',
    'fr': 'fra_Latn',
    'es': 'spa_Latn',
    'de': 'deu_Latn',
    'zh-cn': 'zho_Hans',
    'ja': 'jpn_Jpan',
    'ru': 'rus_Cyrl'
}

# Global variable to store user's detected language
user_lang_code = None


def detect_language(text: str) -> str:
    """
    Detect the ISO language of the input and map it to NLLB code.
    """
    lang = detect(text)
    if lang not in LANG_CODE_MAP:
        raise ValueError(f"Language '{lang}' not supported.")
    return LANG_CODE_MAP[lang]


def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translate text using NLLB model if source and target languages differ.
    """
    if src_lang == tgt_lang:
        return text

    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang)
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


def translate_input_to_english(text: str) -> str:
    """
    Detect user's language and translate input to English. Stores user language globally.
    """
    global user_lang_code
    user_lang_code = detect_language(text)
    return translate(text, user_lang_code, TARGET_LANG_CODE)


def translate_output_to_user_language(text: str) -> str:
    """
    Translate LLM output back to user's original language.
    """
    if user_lang_code is None:
        raise RuntimeError("User language not detected. Call translate_input_to_english first.")
    return translate(text, TARGET_LANG_CODE, user_lang_code)



# Configure logger to log only to the terminal (stdout)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [INGEST] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()]  # only console output
)

logger = logging.getLogger(__name__)


openai_key = os.getenv("OPEN_AI_KEY")
groq_key = os.getenv("GROQ_API_KEY")

# openai_ef = embedding_functions.OpenAIEmbeddingFunction( api_key= openai_key, model_name="text-embedding-3-small")

#new embedding function
# Define a class that conforms to the required interface
local_model = SentenceTransformer("all-MiniLM-L6-v2")

class LocalEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts):
        print("==== Generating local embeddings... ====")
        if isinstance(texts, str):
            texts = [texts]
        return local_model.encode(texts).tolist()

# Instantiate the embedding function
local_ef = LocalEmbeddingFunction()



#below is initializing chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=local_ef)

client = OpenAI(api_key=openai_key)
client2 = Groq(
    api_key=groq_key
)


# response = client2.chat.completions.create(
#     model="llama-3.3-70b-versatile",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is human life expectancy in India?"}
#     ]
# )


# Function to load PDF documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading PDF documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            with fitz.open(pdf_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                documents.append({"id": filename, "text": text})
    return documents


# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# Load documents from the directory
directory_path = "./documents"  # Change this to your PDF folder
documents = load_documents_from_directory(directory_path)

print(f"Loaded {len(documents)} PDF documents")

# Split documents into chunks
chunked_documents = []
for doc in documents:
    print(f"==== Splitting {doc['id']} into chunks ====")
    chunks = split_text(doc["text"])
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

# print(f"Split documents into {len(chunked_documents)} chunks")

# # Function to generate embeddings using OpenAI API
# def get_openai_embedding(text):
#     response = client.embeddings.create(input=text, model="text-embedding-3-small")
#     embedding = response.data[0].embedding
#     print("==== Generating embeddings... ====")
#     return embedding

# Generate embeddings for the document chunks
for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = local_ef(doc["text"])
    
# print(doc["embedding"])



# Upsert documents with embeddings into Chroma
for doc in chunked_documents:
    print("==== Inserting chunks into db;;; ====")
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=doc["embedding"]
    )



# Function to query documents
def query_documents(question, n_results=5):
    # query_embedding = get_openai_embedding(question)
    results = collection.query(query_texts=question, n_results=n_results)

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks


# Function to generate a response from AI
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant from ABB for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Frame your answer in such a way that it is easy understandable by a customer. Always "
        "make sure that answer maintains good company-customer relations. Mention the company as your own wherever required."
        "!!!ANSWER SHOULD BE LESS THAT 1024 TOKENS!!!"
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client2.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        max_tokens=1024,  # Limit the length of the answer
        temperature=0.2,   # Optional: tuning creativity
        top_p=0.9          # Optional: control sampling
    )

    answer = response.choices[0].message.content
    return answer


while True:
    question = input("Enter your question (or type 'exit' to quit): ")
    if question.lower().strip() == 'exit':
        break

    total_start = time.time()

    # Step 1: Translate to English
    t1 = time.time()
    question_translated = translate_input_to_english(question)
    t2 = time.time()
    print("\n[Translated to English]")
    print(question_translated)

    # Step 2: Get Answer
    t3 = time.time()
    relevant_chunks = query_documents(question_translated)
    answer = generate_response(question_translated, relevant_chunks)
    t4 = time.time()
    print("\n[Answer in English]")
    print(answer)

    # Step 3: Translate answer back to user language
    t5 = time.time()
    answer_translated = translate_output_to_user_language(answer)
    t6 = time.time()
    print("\n[Translated Answer]")
    print(answer_translated)

    total_end = time.time()

    # Summary
    print("\n--- Time Summary ---")
    print(f"Input Translation Time: {(t2 - t1)*1000:.2f} ms")
    print(f"Answer Generation Time: {(t4 - t3)*1000:.2f} ms")
    print(f"Output Translation Time: {(t6 - t5)*1000:.2f} ms")
    print(f"Total Time: {(total_end - total_start)*1000:.2f} ms\n")