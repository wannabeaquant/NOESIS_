from sqlalchemy import Column, Integer, String, Boolean
from .base import Base

class AlertSubscriber(Base):
    __tablename__ = 'alert_subscribers'
    user_id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    region_of_interest = Column(String)
    severity_preference = Column(String)
    digest_mode = Column(Boolean, default=False) 