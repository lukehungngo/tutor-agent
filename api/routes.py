from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import tempfile
import os
import uuid
from tempfile import NamedTemporaryFile
from core import DocumentProcessor
from utils import async_time_execution

app = FastAPI(title="Simple Document Processor API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Store processors for different sessions
document_processors = {}


class QueryRequest(BaseModel):
    query: str
    session_id: str
    max_results: int = 4


class SessionInfo(BaseModel):
    session_id: str
    document_name: str
    chunk_count: int


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
