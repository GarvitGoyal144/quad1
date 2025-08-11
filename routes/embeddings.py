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

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Save file temporarily
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Read content (simple text files; for PDFs, integrate pdfminer)
    with open(file_path, "r", encoding="utf-8") as f:
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
