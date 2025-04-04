from fastapi import (
    FastAPI,
)
from fastapi.middleware.cors import CORSMiddleware
from utils import logger
from db import mongo_db
import os
from config.settings import settings
from .auth_routes import router as auth_router
from .essay_routes import router as essay_router
from .quiz_routes import router as quiz_router

# Configure app with extended timeout settings for long-running operations (5-10 minutes)
app = FastAPI(
    title="AI Tutor Document Processor API",
    # Allow requests to run for up to 10 minutes (600 seconds)
    openapi_tags=[
        {
            "name": "API",
            "description": "Endpoints may take 5-10 minutes for processing large documents or complex queries",
        }
    ],
)

# Add CORS middleware with extended timeouts
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include auth routes
app.include_router(auth_router)

app.include_router(essay_router)

app.include_router(quiz_router)


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
