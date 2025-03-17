from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import tempfile
import os
import uuid
import asyncio
from tempfile import NamedTemporaryFile

from tutor_agent.core.document_processor import DocumentProcessor
from tutor_agent.core.vector_store import VectorStoreManager
from tutor_agent.core.question_generator import QuestionGenerator
from tutor_agent.core.exam_session import ExamSession

app = FastAPI(title="AI Tutor Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize components
document_processor = DocumentProcessor()
vector_store = VectorStoreManager()
question_generator = QuestionGenerator()

# Store active exam sessions
exam_sessions = {}

class QuestionRequest(BaseModel):
    query: str
    level: str = "understand"  # Default to understanding level


class UserHighlight(BaseModel):
    text: str
    importance: str = "high"

class UserTag(BaseModel):
    tag: str
    related_terms: List[str] = []

class UserAssistance(BaseModel):
    highlighted_sections: List[UserHighlight] = []
    tags: List[UserTag] = []
    summary: Optional[str] = None

class ExamRequest(BaseModel):
    num_questions: int = 5
    analysis_strategy: str = "holistic"  # "holistic", "chunk-based", or "user-assisted"
    user_assistance: Optional[UserAssistance] = None


class AnswerSubmission(BaseModel):
    session_id: str
    question_idx: int
    answer: str


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Process document
        documents = document_processor.load_document(temp_path)
        chunks = document_processor.split_documents(documents)
        vector_store.create_vector_store(chunks)

        # Clean up
        os.unlink(temp_path)

        return {"message": "Document processed successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-question")
async def generate_question(request: QuestionRequest):
    """Generate a question based on the query and Bloom's level."""
    try:
        # Retrieve relevant context
        relevant_docs = vector_store.similarity_search(request.query)
        context = "\n".join(doc.page_content for doc in relevant_docs)

        # Generate question
        question = question_generator.generate_question(context, request.level)

        return {"question": question}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-all-questions")
async def generate_all_questions(query: str):
    """Generate questions for all Bloom's levels."""
    try:
        # Retrieve relevant context
        relevant_docs = vector_store.similarity_search(query)
        context = "\n".join(doc.page_content for doc in relevant_docs)

        # Generate questions for all levels
        questions = question_generator.generate_question(context, "remember")

        return questions

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start-exam")
async def start_exam(
    file: UploadFile = File(...),
    exam_request: ExamRequest = Depends()
):
    """
    Start a new exam session with the uploaded document.
    
    - **file**: The document file (PDF, TXT, etc.) to use for the exam
    - **exam_request**: Configuration for the exam session
    """
    try:
        # Create new exam session
        session = ExamSession()
        session_id = str(uuid.uuid4())
        
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "document")[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Get user highlights if provided
            user_highlights = None
            if exam_request.analysis_strategy == "user-assisted" and exam_request.user_assistance:
                user_highlights = exam_request.user_assistance.dict()
            
            # Initialize exam with document and options
            questions = await session.initialize_exam(
                document_path=temp_path,
                summary=exam_request.user_assistance.summary if exam_request.user_assistance else None,
                num_questions=exam_request.num_questions,
                analysis_strategy=exam_request.analysis_strategy,
                user_highlights=user_highlights
            )
            
            # Store session
            exam_sessions[session_id] = session
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return {
                "session_id": session_id,
                "questions": questions,
                "analysis_strategy": exam_request.analysis_strategy,
                "message": "Exam created successfully"
            }
        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/exam/{session_id}")
async def get_exam(session_id: str):
    """
    Get the questions for an existing exam session.
    
    - **session_id**: The ID of the exam session
    """
    session = exam_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Exam session not found")
    
    return {
        "session_id": session_id,
        "questions": session.current_questions
    }


@app.post("/submit-answer")
async def submit_answer(submission: AnswerSubmission):
    """
    Submit and evaluate an answer to an exam question.
    
    - **session_id**: The ID of the exam session
    - **question_idx**: The index of the question being answered
    - **answer**: The student's answer to evaluate
    """
    try:
        session = exam_sessions.get(submission.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Exam session not found")
        
        evaluation = await session.evaluate_answer(
            submission.question_idx,
            submission.answer
        )
        
        return {
            "evaluation": evaluation,
            "question": session.current_questions[submission.question_idx]
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-more-questions/{session_id}")
async def generate_more_questions(session_id: str, num_questions: int = 3):
    """
    Generate additional questions for an existing exam session.
    
    - **session_id**: The ID of the exam session
    - **num_questions**: Number of additional questions to generate
    """
    session = exam_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Exam session not found")
    
    try:
        new_questions = await session.generate_exam_questions(num_questions)
        
        return {
            "new_questions": new_questions,
            "all_questions": session.current_questions
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/exam/{session_id}")
async def delete_exam(session_id: str):
    """
    Delete an exam session.
    
    - **session_id**: The ID of the exam session to delete
    """
    if session_id in exam_sessions:
        del exam_sessions[session_id]
        return {"message": "Exam session deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Exam session not found")
