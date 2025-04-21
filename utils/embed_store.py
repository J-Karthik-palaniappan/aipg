# utils/embed_store.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def create_or_load_vectorstore(docs, index_path="faiss_index"):
    embedding = get_embeddings()

    if os.path.exists(index_path):
        print(f"[INFO] Loading existing FAISS index from {index_path}...")
        return FAISS.load_local(index_path, embeddings=embedding, allow_dangerous_deserialization=True)
    
    print("[INFO] Creating new FAISS index...")
    vectorstore = FAISS.from_texts(docs, embedding)
    vectorstore.save_local(index_path)
    return vectorstore

def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
