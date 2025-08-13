from fastapi import APIRouter

router = APIRouter(prefix="/moderate", tags=["moderation"])

@router.post("/flag")
def flag_false_positive():
    # Placeholder: Flag a post/incident as false positive
    return {"status": "flagged"}

@router.post("/confirm")
def confirm_incident():
    # Placeholder: Confirm a high-severity incident
    return {"status": "confirmed"}

@router.post("/merge")
def merge_incidents():
    # Placeholder: Merge duplicate incidents
    return {"status": "merged"} 