from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy import JSON
from .base import Base
import datetime

class RawPost(Base):
    __tablename__ = 'raw_posts'
    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String, index=True)
    content = Column(Text)
    author = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    location_raw = Column(String)
    link = Column(String)
    extra = Column(JSON)  # For any additional metadata 