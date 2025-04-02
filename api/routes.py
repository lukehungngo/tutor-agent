from fastapi import (
    FastAPI,
)
from fastapi.middleware.cors import CORSMiddleware
from utils import logger
from db import mongo_db
import os
from config.settings import settings
from .auth_routes import router as auth_router
from .exam_routes import router as exam_router

app = FastAPI(title="AI Tutor Document Processor API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include auth routes
app.include_router(auth_router)
app.include_router(exam_router)


@app.get("/health")
async def health_check():
    """API health check endpoint."""
    try:
        # Check MongoDB connection
        mongo_health = True
        try:
            mongo_db.db.command("ping")
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
            mongo_health = False

        # Check ChromaDB directory access
        chroma_health = os.access(settings.CHROMA_PERSIST_DIR, os.R_OK | os.W_OK)

        if mongo_health and chroma_health:
            return {"status": "healthy"}
        else:
            return {
                "status": "unhealthy",
                "details": {"mongodb": mongo_health, "chromadb": chroma_health},
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}
