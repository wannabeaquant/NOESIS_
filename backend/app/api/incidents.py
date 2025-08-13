from fastapi import APIRouter, Query, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from app.utils.database import get_db
from app.models.incident import Incident
from app.models.processed_post import ProcessedPost
from app.models.raw_post import RawPost

router = APIRouter(prefix="/incidents", tags=["incidents"])

@router.get("/")
def get_incidents(
    region: Optional[str] = Query(None, description="Filter by region"),
    date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    severity: Optional[str] = Query(None, description="Filter by severity (low/medium/high)"),
    limit: int = Query(50, description="Number of incidents to return"),
    db: Session = Depends(get_db)
):
    """Get incidents with optional filtering"""
    query = db.query(Incident)
    
    if region:
        # Try to match both place names and coordinates
        query = query.filter(
            (Incident.location.contains(region)) |
            (Incident.location_lat.isnot(None) & Incident.location_lng.isnot(None))
        )
    
    if date:
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            next_day = date_obj + timedelta(days=1)
            # Note: This assumes you have a timestamp field in Incident model
            # query = query.filter(Incident.timestamp >= date_obj, Incident.timestamp < next_day)
            pass
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    if severity:
        if severity not in ["low", "medium", "high"]:
            raise HTTPException(status_code=400, detail="Invalid severity. Use low/medium/high")
        query = query.filter(Incident.severity == severity)
    
    incidents = query.order_by(Incident.incident_id.desc()).limit(limit).all()
    
    return [
        {
            "incident_id": incident.incident_id,
            "title": incident.title,
            "description": incident.description,
            "location": incident.location,
            "location_lat": incident.location_lat,
            "location_lng": incident.location_lng,
            "severity": incident.severity,
            "status": incident.status,
            "sources": incident.sources
        }
        for incident in incidents
    ]

@router.get("/latest")
def get_latest_verified(limit: int = Query(10, description="Number of latest incidents"), db: Session = Depends(get_db)):
    """Get latest verified incidents"""
    incidents = db.query(Incident).filter(
        Incident.status.in_(["verified", "medium"])
    ).order_by(Incident.incident_id.desc()).limit(limit).all()
    
    return [
        {
            "incident_id": incident.incident_id,
            "title": incident.title,
            "description": incident.description,
            "location": incident.location,
            "location_lat": incident.location_lat,
            "location_lng": incident.location_lng,
            "severity": incident.severity,
            "status": incident.status,
            "sources": incident.sources
        }
        for incident in incidents
    ]

@router.get("/stats/summary")
def get_incident_stats(db: Session = Depends(get_db)):
    """Get incident statistics"""
    total_incidents = db.query(Incident).count()
    verified_incidents = db.query(Incident).filter(Incident.status == "verified").count()
    high_severity = db.query(Incident).filter(Incident.severity == "high").count()
    
    return {
        "total_incidents": total_incidents,
        "verified_incidents": verified_incidents,
        "high_severity_incidents": high_severity,
        "verification_rate": (verified_incidents / total_incidents * 100) if total_incidents > 0 else 0
    }

@router.get("/dashboard")
def get_dashboard(db: Session = Depends(get_db)):
    """Get real-time dashboard data for frontend"""
    from datetime import datetime, timedelta
    
    # Get basic stats
    total_incidents = db.query(Incident).count()
    verified_incidents = db.query(Incident).filter(Incident.status == "verified").count()
    high_severity = db.query(Incident).filter(Incident.severity == "high").count()
    
    # Get recent activity (last 24 hours)
    yesterday = datetime.now() - timedelta(days=1)
    recent_incidents = db.query(Incident).filter(Incident.incident_id >= total_incidents - 10).all()
    
    # Get severity distribution
    severity_counts = {
        "low": db.query(Incident).filter(Incident.severity == "low").count(),
        "medium": db.query(Incident).filter(Incident.severity == "medium").count(),
        "high": high_severity
    }
    
    # Get status distribution
    status_counts = {
        "unverified": db.query(Incident).filter(Incident.status == "unverified").count(),
        "medium": db.query(Incident).filter(Incident.status == "medium").count(),
        "verified": verified_incidents
    }
    
    # Get top locations (by incident count)
    location_counts = {}
    for incident in db.query(Incident).all():
        location = incident.location or "Unknown"
        location_counts[location] = location_counts.get(location, 0) + 1
    
    top_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        "summary": {
            "total_incidents": total_incidents,
            "verified_incidents": verified_incidents,
            "high_severity_incidents": high_severity,
            "verification_rate": (verified_incidents / total_incidents * 100) if total_incidents > 0 else 0
        },
        "recent_activity": [
            {
                "incident_id": incident.incident_id,
                "title": incident.title,
                "location": incident.location,
                "severity": incident.severity,
                "status": incident.status,
                "sources_count": len(incident.sources) if incident.sources else 0
            }
            for incident in recent_incidents
        ],
        "severity_distribution": severity_counts,
        "status_distribution": status_counts,
        "top_locations": [{"location": loc, "count": count} for loc, count in top_locations],
        "last_updated": datetime.now().isoformat()
    }

@router.get("/{incident_id}")
def get_incident_by_id(incident_id: int, db: Session = Depends(get_db)):
    """Get incident details by ID"""
    incident = db.query(Incident).filter(Incident.incident_id == incident_id).first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    return {
        "incident_id": incident.incident_id,
        "title": incident.title,
        "description": incident.description,
        "location": incident.location,
        "location_lat": incident.location_lat,
        "location_lng": incident.location_lng,
        "severity": incident.severity,
        "status": incident.status,
        "sources": incident.sources
    } 