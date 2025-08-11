from fastapi import FastAPI
from routes.embeddings import router as embeddings_router  # ✅ Correct import

app = FastAPI(title="Document Embedding API with Gemini & pgvector")

# Include routes
app.include_router(embeddings_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Embedding API is running!"}
