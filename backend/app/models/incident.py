from sqlalchemy import Column, Integer, String, Text, Float
from sqlalchemy import JSON
from .base import Base

class Incident(Base):
    __tablename__ = 'incidents'
    incident_id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    description = Column(Text)
    sources = Column(JSON)  # List of source post IDs/links
    location = Column(String)
    location_lat = Column(Float)
    location_lng = Column(Float)
    severity = Column(String)
    status = Column(String) 