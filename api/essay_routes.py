from fastapi import (
    UploadFile,
    File,
    HTTPException,
    Form,
    Depends,
    APIRouter,
)
from pydantic import BaseModel
from typing import Optional
import os
from tempfile import NamedTemporaryFile
from core import (
    DocumentProcessor,
    EssayGenerator,
    Gemma3QuestionGenerator,
    Gemma3AnswerEvaluator,
    AnswerEvaluator,
)
from models import (
    BloomAbstractLevel,
    DocumentInfo,
    User,
    get_temperature_from_bloom_level,
    UserAnswer,
)
from utils import async_time_execution, logger
from db import EssayRepository, ChromaEmbeddingStore
from langchain.schema import Document
from pathlib import Path
from config.settings import settings
from datetime import datetime, timezone
from services import auth_service
import json

router = APIRouter(prefix="/essay", tags=["essay"])

exam_generator = EssayGenerator(Gemma3QuestionGenerator())
answer_evaluator = AnswerEvaluator(Gemma3AnswerEvaluator())

# Initialize MongoDB
essay_repository = EssayRepository()

# Initialize ChromaDB embedding store
embedding_store = ChromaEmbeddingStore(persist_directory=settings.CHROMA_PERSIST_DIR)

# Store processors for different documents (keyed by document_id)
document_processors: dict[str, DocumentProcessor] = {}


class QueryRequest(BaseModel):
    query: str
    document_id: str
    max_results: int = 4


class UploadResponse(BaseModel):
    document_id: str
    document_name: str
    chunk_count: int


class ExamRequest(BaseModel):
    document_id: str
    bloom_level: BloomAbstractLevel
    from_chunk: int
    to_chunk: int


class EvaluateAnswerRequest(BaseModel):
    question_id: str
    answer: str


class SubmitAnswerRequest(BaseModel):
    question_id: str
    answer: str


@router.post("/upload", response_model=UploadResponse)
@async_time_execution
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    user: User = Depends(auth_service.require_auth),
):
    """Upload and process a document.

    Note: Processing large documents may take 5-10 minutes. The system will continue
    processing even if the client connection times out. You can check document status
    by querying the document endpoint.
    """
    try:
        # Save uploaded file temporarily
        with NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename or "document")[1]
        ) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Generate title from filename if not provided
            doc_title = title
            if not doc_title and file.filename:
                # Extract filename without extension
                doc_title = os.path.splitext(file.filename)[0]

            # Create document info object
            doc_info = DocumentInfo(
                user_id=user.id,
                title=doc_title or "Untitled Document",
                filename=file.filename or "document",
                description=description,
                file_size=len(content),
                created_at=datetime.now(timezone.utc),
            )

            # Save document info to MongoDB first to get document_id
            document_id = essay_repository.save_document_info(doc_info)

            # Initialize processor for this document with shared embedding store
            processor = DocumentProcessor(embedding_store=embedding_store)

            # Process document with document_id
            docs = processor.load_document(temp_path, document_id=document_id)
            chunks = processor.process_documents()

            # Update document info with chunk count
            doc_info.chunk_count = len(chunks)

            processor.create_vector_store()

            # Store processor for later use
            document_processors[document_id] = processor

            summary = processor.generate_brief_summary(chunks[:5])

            # Update in DB
            essay_repository.update_document_info(
                document_id, {"chunk_count": len(chunks), "summary": summary}
            )

            # Clean up temporary file
            os.unlink(temp_path)

            return UploadResponse(
                document_id=document_id,
                document_name=file.filename or "document",
                chunk_count=len(chunks),
            )

        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
@async_time_execution
async def query_document(
    request: QueryRequest, user: User = Depends(auth_service.require_auth)
):
    """Query the document."""
    try:
        # Check if processor exists for this document
        if request.document_id not in document_processors:
            # Check if document exists
            doc_info = essay_repository.get_document_info(request.document_id)
            if not doc_info:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document {request.document_id} not found",
                )

            # Create a new processor with the document_id
            processor = DocumentProcessor(embedding_store=embedding_store)
            # Set the document_id for this processor
            processor.document_id = request.document_id
            document_processors[request.document_id] = processor

            # Verify the document embeddings exist in the embedding store
            if not embedding_store.load_document_embeddings(request.document_id):
                raise HTTPException(
                    status_code=404,
                    detail=f"Document data not found for document {request.document_id}",
                )

        processor = document_processors[request.document_id]
        results = processor.similarity_search(request.query, request.max_results)
        return [doc.page_content for doc in results]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summary")
@async_time_execution
async def get_document_summary(
    request: QueryRequest, user: User = Depends(auth_service.require_auth)
):
    """Get the document summary."""
    try:
        # Check if processor exists for this document
        if request.document_id not in document_processors:
            # Check if document exists
            doc_info = essay_repository.get_document_info(request.document_id)
            if not doc_info:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document {request.document_id} not found",
                )

            # Create a new processor with the document_id
            processor = DocumentProcessor(embedding_store=embedding_store)
            processor.document_id = request.document_id
            document_processors[request.document_id] = processor

            # Verify the document embeddings exist in the embedding store
            if not embedding_store.load_document_embeddings(request.document_id):
                raise HTTPException(
                    status_code=404,
                    detail=f"Document data not found for document {request.document_id}",
                )

            # For summary generation, we need to load the documents
            # Get all document chunks from the embedding store
            all_docs = embedding_store.get_all_document_chunks(request.document_id)
            if not all_docs:
                raise HTTPException(
                    status_code=404,
                    detail=f"No document content found for document {request.document_id}",
                )

            # Convert to Document objects for the processor
            processor.documents = [
                Document(page_content=doc["text"], metadata=doc["metadata"])
                for doc in all_docs
            ]

        processor = document_processors[request.document_id]
        summary = processor.get_document_summary()
        return summary
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate_essay")
@async_time_execution
async def generate_essay(
    request: ExamRequest, user: User = Depends(auth_service.require_auth)
):
    """Generate an essay."""
    try:
        # Check if processor exists for this document
        if request.document_id not in document_processors:
            # Check if document exists
            doc_info = essay_repository.get_document_info(request.document_id)
            if not doc_info:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document {request.document_id} not found",
                )

            # Create a new processor with the document_id
            processor = DocumentProcessor(embedding_store=embedding_store)
            processor.document_id = request.document_id
            document_processors[request.document_id] = processor

            # Verify the document embeddings exist in the embedding store
            if not embedding_store.load_document_embeddings(request.document_id):
                raise HTTPException(
                    status_code=404,
                    detail=f"Document data not found for document {request.document_id}",
                )

            # Load document chunks for essay generation
            all_docs = embedding_store.get_all_document_chunks(request.document_id)
            processor.documents = [
                Document(page_content=doc["text"], metadata=doc["metadata"])
                for doc in all_docs
            ]

        processor = document_processors[request.document_id]
        chunks = processor.get_document_chunks(request.from_chunk, request.to_chunk)
        # summary = processor.generate_brief_summary(chunks)
        # logger.info(f"Summary: {summary}")
        summary = "\n\n".join([chunk.page_content for chunk in chunks])

        # Generate essay questions
        result = await exam_generator.generate_essay(
            summary,
            [chunk.page_content for chunk in chunks],
            bloom_level=request.bloom_level,
            temperature=0.3,
        )

        # Set document_id for each question
        for question in result:
            question.document_id = request.document_id
            question.created_at = datetime.now(timezone.utc)
            question.user_id = user.id

        # Save questions to MongoDB
        question_ids = essay_repository.save_questions(result)

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
        logger.error(f"Error generating essay: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate_answer")
@async_time_execution
async def evaluate_answer(
    request: EvaluateAnswerRequest, user: User = Depends(auth_service.require_auth)
):
    """Evaluate an answer to a question."""
    try:
        # Retrieve question from MongoDB
        question_data = essay_repository.get_question(request.question_id)
        if not question_data:
            raise HTTPException(
                status_code=404, detail=f"Question {request.question_id} not found"
            )

        # Evaluate the answer
        context = question_data.context or ""
        question_text = question_data.question
        temperature = get_temperature_from_bloom_level(question_data.bloom_level.value)
        evaluation = await answer_evaluator.evaluate_answer(
            context, question_text, request.answer, temperature
        )

        # Return evaluation results
        return evaluation.as_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/submit_answer")
@async_time_execution
async def submit_answer(
    request: SubmitAnswerRequest, user: User = Depends(auth_service.require_auth)
):
    """Submit and save a user's answer to a question."""
    try:
        assert user.id is not None, "User ID cannot be None"
        # First evaluate the answer
        question_data = essay_repository.get_question(request.question_id)
        if not question_data:
            raise HTTPException(
                status_code=404, detail=f"Question {request.question_id} not found"
            )

        # Evaluate the answer
        context = question_data.context or ""
        question_text = question_data.question
        temperature = get_temperature_from_bloom_level(question_data.bloom_level.value)
        evaluation = await answer_evaluator.evaluate_answer(
            context, question_text, request.answer, temperature
        )

        user_answer = UserAnswer(
            user_id=user.id,
            document_id=question_data.document_id,
            question_id=request.question_id,
            answer_text=request.answer,
            correctness_level=evaluation.correctness_level,
            score=evaluation.score,
            feedback=evaluation.as_dict(),
            improvement_suggestions=evaluation.improvement_suggestions,
            encouragement=evaluation.encouragement,
            next_steps=evaluation.next_steps,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        answer_id = essay_repository.save_answer(user_answer)

        # Return the evaluation with the saved answer ID
        result = user_answer.as_dict()
        result["id"] = answer_id

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str, user: User = Depends(auth_service.require_auth)
):
    """Delete a document and all associated data."""
    try:
        # Check if document exists
        doc_info = essay_repository.get_document_info(document_id)
        if not doc_info:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found",
            )

        # Delete from ChromaDB
        success = embedding_store.delete_document_embeddings(document_id)

        # Remove from document processors if it exists
        if document_id in document_processors:
            del document_processors[document_id]

        # Also delete the document info from MongoDB
        essay_repository.delete_document_info(document_id)

        if success:
            return {"status": "success", "message": f"Document {document_id} deleted"}
        else:
            raise HTTPException(
                status_code=500, detail=f"Failed to delete document {document_id}"
            )
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def get_user_documents(user: User = Depends(auth_service.require_auth)):
    """Get all documents owned by the user."""
    try:
        if not user.id:
            raise HTTPException(status_code=401, detail="User ID is required")

        documents = essay_repository.get_documents_by_user(user.id)
        return {"documents": [doc.to_dict() for doc in documents]}
    except Exception as e:
        logger.error(f"Error retrieving user documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}")
async def get_document_by_id(
    document_id: str, user: User = Depends(auth_service.require_auth)
):
    """Get a document by ID."""
    try:
        document = essay_repository.get_document_info(document_id)
        if not document:
            raise HTTPException(
                status_code=404, detail=f"Document {document_id} not found"
            )
        document_dict = document.to_dict()
        document_dict["questions"] = get_questions_by_document(document_id, user)
        return document_dict
    except Exception as e:
        logger.error(f"Error retrieving document by ID: {e}")


@router.get("/documents/{document_id}/questions")
async def get_questions_for_document(
    document_id: str, user: User = Depends(auth_service.require_auth)
):
    """Get all questions for a specific document."""
    try:
        return get_questions_by_document(document_id, user)
    except Exception as e:
        logger.error(f"Error retrieving questions for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def get_questions_by_document(document_id: str, user: User):
    assert user.id is not None, "User ID cannot be None"
    questions = essay_repository.get_questions_by_document(document_id)
    # Enhance questions with user answers if they exist
    user_answers = essay_repository.get_user_answers_by_document(document_id, user.id)
    enhanced_questions = []
    for question in questions:
        question_dict = question.as_dict()
        # Try to get user's answer for this question if it exists
        # Convert IDs to strings for comparison to ensure type matching
        user_answer = next(
            (
                answer
                for answer in user_answers
                if str(answer.question_id) == str(question.id)
            ),
            None,
        )
        if user_answer:
            question_dict["user_answer"] = user_answer.as_dict()
        enhanced_questions.append(question_dict)

    return enhanced_questions


@router.get("/documents/{document_id}/questions/{question_id}")
async def get_question_by_id(
    document_id: str, question_id: str, user: User = Depends(auth_service.require_auth)
):
    """Get a question by ID."""
    try:
        assert user.id is not None, "User ID cannot be None"
        question = essay_repository.get_question(question_id)
        if not question:
            raise HTTPException(
                status_code=404, detail=f"Question {question_id} not found"
            )
        question_dict = question.as_dict()
        # Enhance question with user answer if it exists
        user_answer = essay_repository.get_user_answer_by_question_id(
            question_id, user.id
        )
        if user_answer:
            question_dict["user_answer"] = user_answer.as_dict()
        return question_dict
    except Exception as e:
        logger.error(f"Error retrieving question by ID: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}/questions/{question_id}")
async def delete_questions(
    question_id: str, user: User = Depends(auth_service.require_auth)
):
    """Delete a question by ID."""
    try:
        assert user.id is not None, "User ID cannot be None"
        question = essay_repository.get_question(question_id)
        if not question:
            raise HTTPException(
                status_code=404, detail=f"Question {question_id} not found"
            )
        if question.user_id != user.id:
            raise HTTPException(
                status_code=403, detail="You are not authorized to delete this question"
            )

        essay_repository.delete_question(question_id)

        return {"status": "success", "message": f"Question {question_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting question: {e}")
        raise HTTPException(status_code=500, detail=str(e))
