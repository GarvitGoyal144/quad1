# create_tables.py
from db import engine, Base
import models

print("📦 Creating tables...")
Base.metadata.create_all(bind=engine)
print("✅ Done!")
