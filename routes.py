from fastapi import APIRouter, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse
import os
import requests
from dotenv import load_dotenv
import tempfile

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

router = APIRouter()

# Helper: extract Gemini generated text from API response
def extract_gemini_text(response_json):
    candidates = response_json.get("candidates")
    if candidates and isinstance(candidates, list) and len(candidates) > 0:
        return candidates[0].get("content", "")
    return ""

@router.post("/hackrx_submission")
async def hackrx_submission(
    doc_url: str = Form(...), 
    questions: str = Form(...)
):
    try:
        # Download document
        response = requests.get(doc_url)
        if response.status_code != 200:
            return JSONResponse(content={"error": "Unable to fetch document"}, status_code=400)

        temp_path = tempfile.mktemp(suffix=".pdf")
        with open(temp_path, "wb") as f:
            f.write(response.content)

        question_list = [q.strip() for q in questions.split("\n") if q.strip()]

        # Build prompt with numbered questions and mention document input
        prompt = "You are an expert assistant analyzing the following document. Answer the questions below:\n\n"
        for i, q in enumerate(question_list, 1):
            prompt += f"{i}. {q}\n"

        # Call Gemini API
        headers = {"Content-Type": "application/json"}
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GOOGLE_API_KEY}"
        payload = {
            "prompt": {
                "text": prompt
            },
            "temperature": 0,
            "maxOutputTokens": 512
        }

        gemini_res = requests.post(gemini_url, json=payload, headers=headers)
        gemini_res.raise_for_status()
        data = gemini_res.json()

        answer_text = extract_gemini_text(data)

        return {"answers": answer_text}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



@router.post("/ask")
async def ask_question(payload: dict = Body(...)):
    try:
        question = payload.get("question")
        if not question:
            return JSONResponse(content={"error": "Missing 'question' in request body"}, status_code=400)

        headers = {"Content-Type": "application/json"}
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GOOGLE_API_KEY}"
        payload = {
            "prompt": {
                "text": question
            },
            "temperature": 0,
            "maxOutputTokens": 512
        }

        gemini_res = requests.post(gemini_url, json=payload, headers=headers)
        gemini_res.raise_for_status()
        data = gemini_res.json()

        answer_text = extract_gemini_text(data)

        return {"answer": answer_text}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
