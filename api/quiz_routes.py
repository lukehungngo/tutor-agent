from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.mc_generator import GoogleGeminiMCGenerator
from models import BloomLevel, TopicMcQuiz, User, MultipleChoiceQuestion
from services import auth_service
from fastapi import Depends
from datetime import datetime
from db import QuizMCRepository
from typing import List, Dict, Any

router = APIRouter(prefix="/quiz", tags=["quiz"])

google_gemini_mc_generator = GoogleGeminiMCGenerator()
topic_quiz_repository = QuizMCRepository()


class GenerateMCQuizRequest(BaseModel):
    topic: str
    description: str


class GenerateMCQuizResponse(BaseModel):
    mc_quiz_id: str
    topic: str
    description: str
    questions: List[Dict[str, Any]]
    correct_answers: Dict[str, str]
    total_score: int


class SubmitAnswerRequest(BaseModel):
    mc_quiz_id: str
    answers: Dict[str, str]


@router.post("/mc_quiz/generate")
async def generate_mc_quiz(
    request: GenerateMCQuizRequest, user: User = Depends(auth_service.require_auth)
):
    questions = []
    remember_questions = await google_gemini_mc_generator.generate_mc_questions(
        request.topic, request.description, BloomLevel.REMEMBER
    )
    questions.extend(remember_questions)
    understand_questions = await google_gemini_mc_generator.generate_mc_questions(
        request.topic, request.description, BloomLevel.UNDERSTAND
    )
    questions.extend(understand_questions)
    apply_questions = await google_gemini_mc_generator.generate_mc_questions(
        request.topic, request.description, BloomLevel.APPLY
    )
    questions.extend(apply_questions)
    analyze_questions = await google_gemini_mc_generator.generate_mc_questions(
        request.topic, request.description, BloomLevel.ANALYZE
    )
    questions.extend(analyze_questions)
    evaluate_questions = await google_gemini_mc_generator.generate_mc_questions(
        request.topic, request.description, BloomLevel.EVALUATE
    )
    questions.extend(evaluate_questions)
    create_questions = await google_gemini_mc_generator.generate_mc_questions(
        request.topic, request.description, BloomLevel.CREATE
    )
    questions.extend(create_questions)

    # Add unique ID to each question
    for i, question in enumerate(questions):
        question.id = str(i)

    topic_mc_quiz = TopicMcQuiz(
        user_id=user.id,
        topic=request.topic,
        description=request.description,
        questions=questions,
        total_questions=len(questions),
        correct_answers={},
        total_score=0,
        created_at=datetime.now(),
        is_completed=False,
        updated_at=datetime.now(),
    )
    mc_quiz_id = topic_quiz_repository.save_mc_quiz(topic_mc_quiz)

    return {
        "id": mc_quiz_id,
        "topic": request.topic,
        "description": request.description,
        "questions": questions,
        "total_questions": len(questions),
        "correct_answers": {},
        "total_score": 0,
        "is_completed": False,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }


@router.get("/mc_quiz")
async def get_mc_quiz_by_user_id(exclude_heavy_data: bool = True, user: User = Depends(auth_service.require_auth)):
    """Get all multiple choice quizzes for the current user."""
    try:
        assert user.id is not None, "User ID is required"
        print(f"Fetching quizzes for user: {user.id}")
        # Get quizzes from repository
        quizzes = topic_quiz_repository.get_mc_quiz_by_user_id(user.id, exclude_heavy_data)
        
        # Convert to dictionary format for JSON response
        quiz_data = [quiz.as_dict() if hasattr(quiz, 'as_dict') else quiz for quiz in quizzes]
        
        return {"quizzes": quiz_data, "count": len(quiz_data)}
    except Exception as e:
        print(f"Error in get_mc_quiz_by_user_id: {str(e)}")
        # Return empty array with error message instead of raising exception
        return {"quizzes": [], "error": str(e), "count": 0}


@router.get("/mc_quiz/{mc_quiz_id}")
async def get_mc_quiz_by_id(mc_quiz_id: str):
    return topic_quiz_repository.get_mc_quiz_by_id(mc_quiz_id)


@router.post("/mc_quiz/submit_answer")
async def submit_answer(request: SubmitAnswerRequest):
    mc_quiz = topic_quiz_repository.get_mc_quiz_by_id(request.mc_quiz_id)
    if not mc_quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    questions = mc_quiz.questions
    if len(questions) != len(request.answers):
        raise HTTPException(status_code=400, detail="Quiz has not been completed")
    for question in questions:
        if question.id not in request.answers:
            raise HTTPException(status_code=400, detail="Answer for question not found")

    for question in questions:
        if question.id in request.answers:
            if question.answer == request.answers[question.id]:
                mc_quiz.total_score += 1
                mc_quiz.correct_answers[question.id] = question.answer
            mc_quiz.chosen_answers[question.id] = request.answers[question.id]
    mc_quiz.updated_at = datetime.now()
    topic_quiz_repository.update_mc_quiz(mc_quiz)

    return {
        "id": mc_quiz.id,
        "topic": mc_quiz.topic,
        "description": mc_quiz.description,
        "questions": questions,
        "total_questions": mc_quiz.total_questions,
        "is_completed": mc_quiz.is_completed,
        "correct_answers": mc_quiz.correct_answers,
        "chosen_answers": mc_quiz.chosen_answers,
        "total_score": mc_quiz.total_score,
        "updated_at": mc_quiz.updated_at,
    }
