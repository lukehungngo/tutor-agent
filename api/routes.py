from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Form,
    Depends,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import tempfile
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
from models.exam import BloomAbstractLevel
from utils import async_time_execution, logger
from config.settings import settings

app = FastAPI(title="Simple Document Processor API")
exam_generator = ExamGenerator(Gemma3QuestionGenerator())
answer_evaluator = AnswerEvaluator(Gemma3AnswerEvaluator())

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


class ExamRequest(BaseModel):
    session_id: str
    bloom_level: BloomAbstractLevel
    from_chunk: int
    to_chunk: int

class EvaluateAnswerRequest(BaseModel):
    session_id: str
    question_id: str
    answer: str

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
            summary,
            [chunk.page_content for chunk in chunks],
            bloom_level=request.bloom_level,
        )
        logger.info(f"Result: {result}")
        return [item.as_dict() for item in result]
    except Exception as e:
        print(e)


@app.post("/evaluate_answer")
@async_time_execution
async def evaluate_answer(request: EvaluateAnswerRequest):
    """Evaluate an answer."""
    try:
        # processor = document_processors[request.session_id]
        # TODO: get question from mongo
        context = """Here's a summary of the provided text focusing solely on the core concepts and ideas:\n\nThe Ethereum protocol utilizes a secure, decentralized ledger technology – version 11 – designed to create and maintain accounts without requiring traditional funds transfers.  Key features include “σ′, “g′,” and “A‶ states representing the current state, pending gas, and accumulated substates, respectively. These states allow for precise tracking of account activity, crucial for verifying transactions and ensuring security.  Messages involve complex calculations utilizing cryptographic hashes and require careful management of gas consumption. Errors within the execution process necessitate reverting the account to a safe state, preventing irreversible loss of assets. Essentially, Ethereum provides a robust mechanism for recording and validating digital asset transactions while maintaining decentralization and immutability"""
        question = "Explain the role of the gas cost in the EVM, and how it relates to the execution of a transaction."
        student_answer = "The gas cost represents the computational effort required to execute the transaction. It’s proportional to the size of the operation and the complexity of the state changes, and is paid for on a just-in-time basis"
        evaluation = await answer_evaluator.evaluate_answer(context, question, student_answer)
        return evaluation
    except Exception as e:
        print(e)