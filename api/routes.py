from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import tempfile
import os
import uuid
from tempfile import NamedTemporaryFile
from core import DocumentProcessor, ExamGenerator, Gemma3QuestionGenerator, BloomAbstractLevel
from utils import async_time_execution, logger

app = FastAPI(title="Simple Document Processor API")
gemma3_question_generator = Gemma3QuestionGenerator()
exam_generator = ExamGenerator(gemma3_question_generator)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Store processors for different sessions
document_processors: dict[str, DocumentProcessor] = {}

# Session management
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    logger.info("Application shutting down, cleaning up resources...")
    
    # Clean up document processors
    for session_id, processor in list(document_processors.items()):
        await cleanup_session(session_id)
        
    # Clean up global models
    if gemma3_question_generator:
        gemma3_question_generator.cleanup()
        
    logger.info("Resource cleanup complete")

async def cleanup_session(session_id: str):
    """Clean up resources for a specific session."""
    if session_id in document_processors:
        logger.info(f"Cleaning up session {session_id}")
        try:
            processor = document_processors[session_id]
            # Add any specific cleanup for DocumentProcessor if needed
            del document_processors[session_id]
            logger.info(f"Session {session_id} cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")

class QueryRequest(BaseModel):
    query: str
    session_id: str
    max_results: int = 4


class SessionInfo(BaseModel):
    session_id: str
    document_name: str
    chunk_count: int

class ExamRequest(BaseModel):
    session_id: str
    bloom_level: BloomAbstractLevel
    from_chunk: int
    to_chunk: int


@app.post("/upload", response_model=SessionInfo)
@async_time_execution
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    try:
        # Create session ID
        session_id = str(uuid.uuid4())

        # Save uploaded file temporarily
        with NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename or "document")[1]
        ) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Initialize processor for this session
            processor = DocumentProcessor()

            # Process document
            docs = processor.load_document(temp_path)
            chunks = processor.process_documents()
            processor.create_vector_store()

            # Store processor for later use
            document_processors[session_id] = processor

            # Clean up temporary file
            os.unlink(temp_path)

            return SessionInfo(
                session_id=session_id,
                document_name=file.filename or "document",
                chunk_count=len(chunks),
            )

        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
@async_time_execution
async def query_document(request: QueryRequest):
    """Query the document."""
    try:
        processor = document_processors[request.session_id]
        results = processor.similarity_search(request.query, request.max_results)
        return [doc.page_content for doc in results]
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summary")
@async_time_execution
async def get_document_summary(request: QueryRequest):
    """Get the document summary."""
    try:
        processor = document_processors[request.session_id]
        return processor.get_document_summary()
    except Exception as e:
        print(e)


@app.post("/generate_exam")
@async_time_execution
async def generate_exam(request: ExamRequest):
    """Generate an exam."""
    try:
        processor = document_processors[request.session_id]
        chunks = processor.get_document_chunks(request.from_chunk, request.to_chunk)
        summary = processor.generate_brief_summary(chunks)
        logger.info(f"Summary: {summary}")
        result = await exam_generator.generate_exam(
            summary, [chunk.page_content for chunk in chunks], bloom_level=request.bloom_level
        )
        logger.info(f"Result: {result}")
        return [
            item.as_dict()
            for item in result
        ]
    except Exception as e:
        print(e)

@app.post("/close_session")
@async_time_execution
async def close_session(session_id: str):
    """Close a session and clean up its resources."""
    if session_id not in document_processors:
        raise HTTPException(status_code=404, detail="Session not found")
    
    await cleanup_session(session_id)
    return {"status": "success", "message": f"Session {session_id} closed"}
