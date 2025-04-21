# utils/text_processing.py

import fitz  # PyMuPDF
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def extract_text_from_pdf(file_path):
    print(f"[INFO] Reading PDF: {file_path}")
    doc = fitz.open(file_path)
    text = ""
    for page in tqdm(doc, desc="Extracting pages"):
        text += page.get_text()
    doc.close()
    return text

def clean_and_chunk_text(text):
    # Basic cleaning
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    print("[INFO] Splitting text into chunks...")
    chunks = splitter.split_text(text)

    print(f"[INFO] Total chunks created: {len(chunks)}")
    return chunks

def load_pdf_and_prepare_chunks(pdf_path):
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = clean_and_chunk_text(raw_text)
    return chunks

if __name__ == "__main__":
    chunks = load_pdf_and_prepare_chunks("../data.pdf")
    # for i in chunks[:3]:
        # print(i,"\n\n")