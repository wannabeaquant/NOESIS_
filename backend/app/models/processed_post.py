from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy import JSON
from .base import Base
import datetime

class ProcessedPost(Base):
    __tablename__ = 'processed_posts'
    id = Column(Integer, primary_key=True, index=True)
    raw_post_id = Column(Integer, index=True)
    protest_score = Column(Float)
    sentiment_score = Column(Float)
    location_lat = Column(Float)
    location_lng = Column(Float)
    language = Column(String)
    platform = Column(String)
    link = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    entities = Column(JSON)  # NER results
    status = Column(String, default='unverified') 