from fastapi import FastAPI
from routes import embeddings

app = FastAPI(title="Document Embedding API with Gemini & pgvector")

# Include routes
app.include_router(embeddings.router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Embedding API is running!"}
