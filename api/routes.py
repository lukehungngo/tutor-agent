from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import uuid
from tempfile import NamedTemporaryFile
from core import (
    DocumentProcessor,
    ExamGenerator,
    Gemma3QuestionGenerator,
    Gemma3AnswerEvaluator,
    AnswerEvaluator,
)
from models import BloomAbstractLevel, Question, DocumentInfo
from utils import async_time_execution, logger
from db.mongo import MongoDB
from db.chroma_embedding import ChromaEmbeddingStore
from langchain.schema import Document
from pathlib import Path
from config.settings import settings
from datetime import datetime, timezone
import time

app = FastAPI(title="AI Tutor Document Processor API")
exam_generator = ExamGenerator(Gemma3QuestionGenerator())
answer_evaluator = AnswerEvaluator(Gemma3AnswerEvaluator())

# Initialize MongoDB
mongo_db = MongoDB()

# Initialize ChromaDB embedding store
embedding_store = ChromaEmbeddingStore(persist_directory=settings.chroma_persist_dir)

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


class QueryRequest(BaseModel):
    query: str
    session_id: str
    max_results: int = 4


class SessionInfo(BaseModel):
    session_id: str
    document_name: str
    chunk_count: int
    document_id: str

class ExamRequest(BaseModel):
    session_id: str
    bloom_level: BloomAbstractLevel
    from_chunk: int
    to_chunk: int


class EvaluateAnswerRequest(BaseModel):
    question_id: str
    answer: str


class SubmitAnswerRequest(BaseModel):
    question_id: str
    user_id: str
    answer: str


@app.post("/upload", response_model=SessionInfo)
@async_time_execution
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    """Upload and process a document."""
    start_time = time.time()
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
            # Initialize processor for this session with shared embedding store
            processor = DocumentProcessor(embedding_store=embedding_store)

            # Process document
            docs = processor.load_document(temp_path, session_id=session_id)
            chunks = processor.process_documents()
            processor.create_vector_store()

            # Store processor for later use
            document_processors[session_id] = processor

            # Clean up temporary file
            os.unlink(temp_path)

            # Generate title from filename if not provided
            doc_title = title
            if not doc_title and file.filename:
                # Extract filename without extension
                doc_title = os.path.splitext(file.filename)[0]
            
            # Create document info object
            doc_info = DocumentInfo(
                title=doc_title or "Untitled Document",
                filename=file.filename or "document",
                session_id=session_id,
                description=description,
                file_size=len(content),
                chunk_count=len(chunks),
                created_at=datetime.now(timezone.utc),
            )

            # Save document info to MongoDB
            document_id = mongo_db.save_document_info(doc_info)

            return SessionInfo(
                session_id=session_id,
                document_name=file.filename or "document",
                chunk_count=len(chunks),
                document_id=document_id,
            )

        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
@async_time_execution
async def query_document(request: QueryRequest):
    """Query the document."""
    try:
        # Check if processor exists for this session
        if request.session_id not in document_processors:
            # Create a new processor with the existing session
            processor = DocumentProcessor(embedding_store=embedding_store)
            # Set the session ID for this processor
            processor.session_id = request.session_id
            document_processors[request.session_id] = processor
            
            # Verify the session exists in the embedding store
            if not embedding_store.load_session(request.session_id):
                raise HTTPException(
                    status_code=404, 
                    detail=f"Session {request.session_id} not found or could not be loaded"
                )
        
        processor = document_processors[request.session_id]
        results = processor.similarity_search(request.query, request.max_results)
        return [doc.page_content for doc in results]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summary")
@async_time_execution
async def get_document_summary(request: QueryRequest):
    """Get the document summary."""
    try:
        # Check if processor exists for this session
        if request.session_id not in document_processors:
            # Create a new processor with the existing session
            processor = DocumentProcessor(embedding_store=embedding_store)
            processor.session_id = request.session_id
            document_processors[request.session_id] = processor
            
            # Verify the session exists in the embedding store
            if not embedding_store.load_session(request.session_id):
                raise HTTPException(
                    status_code=404, 
                    detail=f"Session {request.session_id} not found or could not be loaded"
                )
                
            # For summary generation, we need to load the documents
            # Get all documents from the embedding store
            all_docs = embedding_store.get_all_documents(request.session_id)
            if not all_docs:
                raise HTTPException(
                    status_code=404,
                    detail=f"No documents found for session {request.session_id}"
                )
            
            # Convert to Document objects for the processor
            processor.documents = [
                Document(
                    page_content=doc["text"],
                    metadata=doc["metadata"]
                )
                for doc in all_docs
            ]
        
        processor = document_processors[request.session_id]
        summary = processor.get_document_summary()
        return summary
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_exam")
@async_time_execution
async def generate_exam(request: ExamRequest):
    """Generate an exam."""
    try:
        # Check if processor exists for this session
        if request.session_id not in document_processors:
            # Create a new processor with the existing session
            processor = DocumentProcessor(embedding_store=embedding_store)
            processor.session_id = request.session_id
            document_processors[request.session_id] = processor
            
            # Verify the session exists in the embedding store
            if not embedding_store.load_session(request.session_id):
                raise HTTPException(
                    status_code=404, 
                    detail=f"Session {request.session_id} not found or could not be loaded"
                )
                
            # Load document chunks for exam generation
            all_docs = embedding_store.get_all_documents(request.session_id)
            processor.documents = [
                Document(
                    page_content=doc["text"],
                    metadata=doc["metadata"]
                )
                for doc in all_docs
            ]
        
        processor = document_processors[request.session_id]
        chunks = processor.get_document_chunks(request.from_chunk, request.to_chunk)
        summary = processor.generate_brief_summary(chunks)
        logger.info(f"Summary: {summary}")
        
        result = await exam_generator.generate_exam(
            summary,
            [chunk.page_content for chunk in chunks],
            bloom_level=request.bloom_level,
        )
        
        # Save questions to MongoDB
        question_ids = mongo_db.save_questions(result)
        
        # Return questions with IDs
        questions_with_ids = []
        for i, question in enumerate(result):
            question_dict = question.as_dict()
            if i < len(question_ids):
                question_dict["id"] = question_ids[i]
            questions_with_ids.append(question_dict)
            
        return questions_with_ids
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating exam: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate_answer")
@async_time_execution
async def evaluate_answer(request: EvaluateAnswerRequest):
    """Evaluate an answer to a question."""
    try:
        # Retrieve question from MongoDB
        question_data = mongo_db.get_question(request.question_id)
        if not question_data:
            raise HTTPException(
                status_code=404,
                detail=f"Question {request.question_id} not found"
            )
        
        # Evaluate the answer
        context = question_data.get("context", "")
        question_text = question_data.get("question", "")
        evaluation = await answer_evaluator.evaluate_answer(
            context, 
            question_text, 
            request.answer
        )
        
        # Return evaluation results
        return evaluation.as_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/submit_answer")
@async_time_execution
async def submit_answer(request: SubmitAnswerRequest):
    """Submit and save a user's answer to a question."""
    try:
        # First evaluate the answer
        question_data = mongo_db.get_question(request.question_id)
        if not question_data:
            raise HTTPException(
                status_code=404,
                detail=f"Question {request.question_id} not found"
            )
        
        # Evaluate the answer
        context = question_data.get("context", "")
        question_text = question_data.get("question", "")
        evaluation = await answer_evaluator.evaluate_answer(
            context, 
            question_text, 
            request.answer
        )
        
        # Calculate score based on correctness level
        score = None
        if evaluation.correctness_level.value == "correct":
            score = 100
        elif evaluation.correctness_level.value == "partially_correct":
            score = 50
        elif evaluation.correctness_level.value == "incorrect":
            score = 0
        
        # Save the answer to MongoDB
        answer_id = mongo_db.save_answer(
            document_id=request.document_id,
            question_id=request.question_id,
            user_id=request.user_id,
            answer_text=request.answer,
            score=score,
            feedback=str(evaluation.as_dict())
        )
        
        # Return the evaluation with the saved answer ID
        result = evaluation.as_dict()
        result["answer_id"] = answer_id
        result["score"] = score
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions():
    """List all available sessions."""
    try:
        # This will list all unique session IDs from the MongoDB collection
        sessions = embedding_store.collection.distinct("metadata.session_id")
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all associated data."""
    try:
        # Delete from ChromaDB and MongoDB
        success = embedding_store.delete_session(session_id)
        
        # Remove from document processors if it exists
        if session_id in document_processors:
            del document_processors[session_id]
            
        if success:
            return {"status": "success", "message": f"Session {session_id} deleted"}
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete session {session_id}"
            )
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        chroma_health = os.access(settings.chroma_persist_dir, os.R_OK | os.W_OK)
        
        if mongo_health and chroma_health:
            return {"status": "healthy"}
        else:
            return {
                "status": "unhealthy", 
                "details": {
                    "mongodb": mongo_health,
                    "chromadb": chroma_health
                }
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}