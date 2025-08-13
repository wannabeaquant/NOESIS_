from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, Response

app = FastAPI(
    title="NOESIS Backend",
    description="Real-time OSINT Civil Unrest Detection System",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

def rate_limit_exceeded_handler(request: Request, exc: Exception) -> Response:
    # type: ignore[reportArgumentType]
    from slowapi.errors import RateLimitExceeded
    if isinstance(exc, RateLimitExceeded):
        return _rate_limit_exceeded_handler(request, exc)
    raise exc

app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Include API routers
from app.api import router as api_router
app.include_router(api_router)

@app.get("/")
def root():
    return {
        "message": "NOESIS Backend is running",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
def health_check():
    """Health check endpoint for Docker"""
    return {"status": "healthy", "service": "NOESIS Backend"} 