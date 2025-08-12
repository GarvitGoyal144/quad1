from fastapi import FastAPI
from main_routes import router as main_router   # This is your routes.py
from routes.embeddings import router as embeddings_router
import os
import uvicorn

app = FastAPI(title="Document Embedding API with Gemini & pgvector")

# Include routes
app.include_router(main_router)
app.include_router(embeddings_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Embedding API is running!"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Railway sets PORT dynamically
    uvicorn.run("main:app", host="0.0.0.0", port=port)
