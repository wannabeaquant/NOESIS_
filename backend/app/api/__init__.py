from fastapi import APIRouter
from .incidents import router as incidents_router
from .alerts import router as alerts_router
from .moderation import router as moderation_router
from .collection import router as collection_router
from .predictions import router as predictions_router

router = APIRouter()
router.include_router(incidents_router)
router.include_router(alerts_router)
router.include_router(moderation_router)
router.include_router(collection_router) 
router.include_router(predictions_router) 