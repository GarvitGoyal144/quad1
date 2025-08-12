import os
import tempfile
import shutil
import requests
import fitz  # PyMuPDF
import docx
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from db import get_db
from models import DocumentEmbedding

# Load environment variables
load_dotenv()

router = APIRouter()

# Gemini API config
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"

# ---------- Gemini Embedding ----------
def get_embedding_from_gemini(text: str):
    payload = {
        "model": "models/text-embedding-004",
        "content": {"parts": [{"text": text}]}
    }
    response = requests.post(
        f"{GEMINI_EMBED_URL}?key={GEMINI_API_KEY}",
        json=payload
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Gemini API error: {response.text}"
        )

    data = response.json()
    try:
        return data["embedding"]["values"]
    except KeyError:
        raise HTTPException(status_code=500, detail="Invalid Gemini API response format.")

# ---------- File Parsing ----------
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_email(file_path):
    import email
    with open(file_path, "rb") as f:
        msg = email.message_from_binary_file(f)
    text_parts = []
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            text_parts.append(part.get_payload(decode=True).decode(errors="ignore"))
    return "\n".join(text_parts)

# ---------- Chunking ----------
def chunk_text(text, chunk_size=3000):
    """Split text into smaller chunks to avoid hitting API limits."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ---------- Upload & Embed ----------
@router.post("/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Save file temporarily
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract text based on file type
    filename_lower = file.filename.lower()
    if filename_lower.endswith(".pdf"):
        content = extract_text_from_pdf(file_path)
    elif filename_lower.endswith(".docx"):
        content = extract_text_from_docx(file_path)
    elif filename_lower.endswith(".eml"):
        content = extract_text_from_email(file_path)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

    if not content.strip():
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="No readable text found in file.")

    # Split into chunks
    chunks = chunk_text(content)

    all_embeddings = []
    for chunk in chunks:
        emb = get_embedding_from_gemini(chunk)
        all_embeddings.append(emb)

    # Store embeddings in DB
    for i, emb in enumerate(all_embeddings):
        doc = DocumentEmbedding(
            filename=f"{file.filename} - chunk {i+1}",
            content=chunks[i],
            embedding=emb
        )
        db.add(doc)
    db.commit()

    # Clean up temp files
    shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "message": "File uploaded and embedded successfully",
        "total_chunks": len(chunks)
    }
