from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List
from modules import ingest, chunk_ember, vector_store, llm_reasoning

router = APIRouter()

class QueryRequest(BaseModel):
    document_url: str
    questions: List[str]

@router.post("/hackrx/run")
async def run_query(req: QueryRequest):
    try:
        # 1. Ingest document (download, parse text)
        raw_text, doc_meta = ingest.load_and_parse(req.document_url)

        # 2. Chunk and Embed
        chunks = chunk_ember.split_into_chunks(raw_text)
        vector_store.insert_chunks(doc_meta['doc_id'], chunks)

        # 3. Process questions
        responses = []
        for question in req.questions:
            result = llm_reasoning.answer_query(question, doc_meta['doc_id'])
            responses.append(result)

        return { "answers": responses }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
