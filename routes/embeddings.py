import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session
from db import get_db
from models import DocumentEmbedding
from dotenv import load_dotenv
import requests

load_dotenv()

router = APIRouter()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"

def get_embedding_from_gemini(text: str):
    payload = {
        "model": "models/text-embedding-004",
        "content": {"parts": [{"text": text}]}
    }
    response = requests.post(
        f"{GEMINI_EMBED_URL}?key={GEMINI_API_KEY}",
        json=payload
    )
    response.raise_for_status()
    data = response.json()
    return data["embedding"]["values"]

import fitz  # PyMuPDF
import docx

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


@router.post("/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Save file temporarily
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Detect and extract text based on file type
    filename_lower = file.filename.lower()
    if filename_lower.endswith(".pdf"):
        content = extract_text_from_pdf(file_path)
    elif filename_lower.endswith(".docx"):
        content = extract_text_from_docx(file_path)
    elif filename_lower.endswith(".eml"):
        content = extract_text_from_email(file_path)
    else:  # Default: try reading as plain text
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

    # Get embedding
    embedding = get_embedding_from_gemini(content)

    # Store in database
    doc = DocumentEmbedding(
        filename=file.filename,
        content=content,
        embedding=embedding
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    return {"message": "File uploaded and embedded successfully", "id": doc.id}