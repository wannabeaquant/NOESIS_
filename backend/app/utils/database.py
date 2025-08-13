from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from app.models.base import Base
# Import all models to ensure they're registered with SQLAlchemy
from app.models.raw_post import RawPost
from app.models.processed_post import ProcessedPost
from app.models.incident import Incident
from app.models.alert_subscriber import AlertSubscriber
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import asyncio

load_dotenv()

# Use SQLite instead of PostgreSQL - much simpler setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./noesis.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@asynccontextmanager
async def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        await asyncio.sleep(0)  # allow async context, even if not needed
        db.close()

def create_tables():
    Base.metadata.create_all(bind=engine) 