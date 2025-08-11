from sqlalchemy import Column, Integer, String
from pgvector.sqlalchemy import Vector
from db import Base

class DocumentEmbedding(Base):
    __tablename__ = "document_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    content = Column(String, nullable=False)
    embedding = Column(Vector(768))  # Dimension depends on the model output
