from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api.routes import router as api_router

app = FastAPI(
    title="Federated Transfer Learning - Fault Diagnosis System",
    description="A federated transfer learning system for fault diagnosis across multiple factory devices",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": "Federated Transfer Learning - Fault Diagnosis System",
        "version": "1.0.0",
        "endpoints": {
            "api_docs": "/docs",
            "api_prefix": "/api/v1"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
