from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.utils.database import get_db
from app.services.data_orchestrator import DataOrchestrator
from app.models.raw_post import RawPost
from app.models.processed_post import ProcessedPost
from app.models.incident import Incident

router = APIRouter(prefix="/collection", tags=["collection"])

@router.post("/run-cycle")
def run_collection_cycle(db: Session = Depends(get_db)):
    """Manually trigger a data collection and processing cycle"""
    try:
        orchestrator = DataOrchestrator(db)
        results = orchestrator.run_collection_cycle()
        
        return {
            "status": "success",
            "message": "Data collection cycle completed successfully",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running collection cycle: {str(e)}")

@router.get("/status")
def get_collection_status(db: Session = Depends(get_db)):
    """Get status of data collection system"""
    try:
        # Get counts from database
        raw_count = db.query(RawPost).count()
        processed_count = db.query(ProcessedPost).count()
        incident_count = db.query(Incident).count()
        
        return {
            "raw_posts": raw_count,
            "processed_posts": processed_count,
            "incidents": incident_count,
            "status": "operational"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        } 