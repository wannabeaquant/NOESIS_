from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from app.utils.database import get_db
from app.models.alert_subscriber import AlertSubscriber

router = APIRouter(prefix="/alerts", tags=["alerts"])

class AlertSubscription(BaseModel):
    email: str
    region_of_interest: Optional[str] = None
    severity_preference: Optional[str] = "medium"  # low, medium, high
    digest_mode: Optional[bool] = False

class AlertResponse(BaseModel):
    status: str
    message: str
    subscriber_id: Optional[int] = None

@router.post("/subscribe", response_model=AlertResponse)
def subscribe(subscription: AlertSubscription, db: Session = Depends(get_db)):
    """Subscribe to real-time alerts"""
    try:
        # Check if already subscribed
        existing = db.query(AlertSubscriber).filter(AlertSubscriber.email == subscription.email).first()
        if existing:
            return AlertResponse(
                status="already_subscribed",
                message="Email already subscribed to alerts",
                subscriber_id=existing.user_id
            )
        
        # Create new subscriber
        subscriber = AlertSubscriber(
            email=subscription.email,
            region_of_interest=subscription.region_of_interest,
            severity_preference=subscription.severity_preference,
            digest_mode=subscription.digest_mode
        )
        db.add(subscriber)
        db.commit()
        db.refresh(subscriber)
        
        return AlertResponse(
            status="subscribed",
            message="Successfully subscribed to NOESIS alerts!",
            subscriber_id=subscriber.user_id
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to subscribe: {str(e)}")

@router.post("/unsubscribe", response_model=AlertResponse)
def unsubscribe(subscription: AlertSubscription, db: Session = Depends(get_db)):
    """Unsubscribe from alerts"""
    try:
        subscriber = db.query(AlertSubscriber).filter(AlertSubscriber.email == subscription.email).first()
        if not subscriber:
            return AlertResponse(
                status="not_found",
                message="Email not found in subscribers"
            )
        
        db.delete(subscriber)
        db.commit()
        
        return AlertResponse(
            status="unsubscribed",
            message="Successfully unsubscribed from NOESIS alerts"
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to unsubscribe: {str(e)}")

@router.get("/subscribers")
def get_subscribers(db: Session = Depends(get_db)):
    """Get all alert subscribers (admin endpoint)"""
    subscribers = db.query(AlertSubscriber).all()
    return [
        {
            "user_id": sub.user_id,
            "email": sub.email,
            "region_of_interest": sub.region_of_interest,
            "severity_preference": sub.severity_preference,
            "digest_mode": sub.digest_mode
        }
        for sub in subscribers
    ] 