from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NOESIS Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
from app.api import router as api_router
app.include_router(api_router)

@app.get("/")
def root():
    return {"message": "NOESIS Backend is running."}

@app.get("/health")
def health():
    return {"status": "healthy", "message": "NOESIS Backend is operational"} 