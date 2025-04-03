# TODO: Move all models of core to this file

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
from utils import logger
from datetime import datetime


class BloomLevel(Enum):
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class BloomAbstractLevel(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


def get_bloom_level_abstract(bloom_level: str) -> BloomAbstractLevel:
    if (
        bloom_level == BloomLevel.REMEMBER.value
        or bloom_level == BloomLevel.UNDERSTAND.value
    ):
        return BloomAbstractLevel.BASIC
    elif (
        bloom_level == BloomLevel.APPLY.value or bloom_level == BloomLevel.ANALYZE.value
    ):
        return BloomAbstractLevel.INTERMEDIATE
    elif (
        bloom_level == BloomLevel.EVALUATE.value
        or bloom_level == BloomLevel.CREATE.value
    ):
        return BloomAbstractLevel.ADVANCED
    else:
        raise ValueError(f"Invalid bloom level: {bloom_level}")


def get_temperature_from_bloom_level(bloom_level: str) -> float:
    if (
        bloom_level == BloomLevel.REMEMBER.value
        or bloom_level == BloomLevel.UNDERSTAND.value
    ):
        return 0.1
    elif (
        bloom_level == BloomLevel.APPLY.value or bloom_level == BloomLevel.ANALYZE.value
    ):
        return 0.2
    else:
        return 0.3


@dataclass
class Question:
    id: Optional[str] = None
    user_id: Optional[str] = None
    question: str = ""
    bloom_level: BloomLevel = BloomLevel.REMEMBER
    document_id: str = ""
    context: Optional[str] = None
    hint: Optional[str] = None
    answer: Optional[str] = None
    created_at: Optional[datetime] = None

    @staticmethod
    def from_dict(data: Dict) -> "Question":
        bloom_level_data = data.get("bloom_level", BloomLevel.REMEMBER.value)

        # If bloom_level is already a BloomLevel enum, use it directly
        if isinstance(bloom_level_data, BloomLevel):
            bloom_level = bloom_level_data
        else:
            # Otherwise, try to convert string to BloomLevel enum
            try:
                bloom_level = BloomLevel(bloom_level_data)
            except (ValueError, TypeError):
                # Default to REMEMBER if conversion fails
                logger.warning(
                    f"Invalid bloom_level value: {bloom_level_data}, defaulting to REMEMBER"
                )
                bloom_level = BloomLevel.REMEMBER

        return Question(
            id=data.get("_id") or data.get("id"),
            user_id=data.get("user_id", None),
            question=data["question"],
            bloom_level=bloom_level,
            hint=data.get("hint"),
            answer=data.get("answer"),
            document_id=data.get("document_id", ""),
            context=data.get("context"),
            created_at=data.get("created_at"),
        )

    def as_dict(self) -> Dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "question": self.question,
            "bloom_level": self.bloom_level.value,
            "hint": self.hint,
            "answer": self.answer,
            "context": self.context,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_raw_response(response: Dict, context: str) -> List["Question"]:
        """Convert a raw LLM response into a list of Question objects.

        Args:
            response: Raw dictionary response from LLM containing questions
            context: The context to associate with the questions

        Returns:
            List of Question objects
        """
        questions = []
        if not isinstance(response, Dict) or "questions" not in response:
            return questions

        for raw in response["questions"]:
            try:
                question = Question.from_dict(raw)
                question.context = context
                questions.append(question)
            except (KeyError, ValueError) as e:
                logger.error(f"Error processing question: {e}")
                continue

        return questions


class CorrectnessLevel(Enum):
    """Enum representing the correctness level of an answer."""

    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"


@dataclass
class EvaluationResult:
    """Data class for storing the result of an answer evaluation."""

    correctness_level: CorrectnessLevel
    score: int
    accurate_parts: List[str]
    improvement_suggestions: List[str]
    encouragement: str
    next_steps: str

    @staticmethod
    def from_dict(data: Dict) -> "EvaluationResult":
        """
        Create an EvaluationResult from a dictionary.

        Args:
            data: Dictionary containing evaluation data

        Returns:
            EvaluationResult object
        """
        try:
            correctness_level = CorrectnessLevel(
                data.get("correctness_level", "partially_correct")
            )
        except (ValueError, KeyError):
            correctness_level = CorrectnessLevel.PARTIALLY_CORRECT

        return EvaluationResult(
            correctness_level=correctness_level,
            score=data.get("score", 0),
            accurate_parts=data.get("accurate_parts", []),
            improvement_suggestions=data.get("improvement_suggestions", []),
            encouragement=data.get("encouragement", "Good effort!"),
            next_steps=data.get("next_steps", "Review the related concepts."),
        )

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the evaluation result to a dictionary.

        Returns:
            Dictionary representation of the evaluation result
        """
        return {
            "correctness_level": self.correctness_level.value,
            "score": self.score,
            "accurate_parts": self.accurate_parts,
            "improvement_suggestions": self.improvement_suggestions,
            "encouragement": self.encouragement,
            "next_steps": self.next_steps,
        }

    @staticmethod
    def from_raw_response(response: Dict) -> "EvaluationResult":
        """
        Convert a raw LLM response into an EvaluationResult object.

        Args:
            response: Raw dictionary response from LLM containing evaluation

        Returns:
            EvaluationResult object
        """
        try:
            return EvaluationResult.from_dict(response)
        except Exception as e:
            logger.error(f"Error processing evaluation response: {e}")
            # Return a default evaluation in case of error
            return EvaluationResult(
                correctness_level=CorrectnessLevel.PARTIALLY_CORRECT,
                score=0,
                accurate_parts=[],
                improvement_suggestions=["Please try again"],
                encouragement="Thank you for your submission.",
                next_steps="Please try again or contact support if the issue persists.",
            )

@dataclass
class UserAnswer:
    id: Optional[str] = None
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    question_id: Optional[str] = None
    answer_text: Optional[str] = None
    correctness_level: Optional[CorrectnessLevel] = None
    score: Optional[int] = None
    feedback: Optional[str] = None
    improvement_suggestions: Optional[List[str]] = None
    encouragement: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    
    def as_dict(self) -> Dict[str, Any]:
        assert self.correctness_level is not None, "Correctness level cannot be None"
        return {
            "id": self.id,
            "user_id": self.user_id,
            "document_id": self.document_id,
            "question_id": self.question_id,
            "answer_text": self.answer_text,
            "correctness_level": self.correctness_level.value,
            "score": self.score,
            "feedback": self.feedback,
            "improvement_suggestions": self.improvement_suggestions,
            "encouragement": self.encouragement,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(data: Dict) -> "UserAnswer":
        assert data.get("correctness_level") is not None, "Correctness level cannot be None"
        return UserAnswer(
            id=data.get("_id") or data.get("id"),
            user_id=data.get("user_id"),
            document_id=data.get("document_id"),
            question_id=data.get("question_id"),
            answer_text=data.get("answer_text"),
            correctness_level=CorrectnessLevel(data.get("correctness_level")),
            score=data.get("score"),
            feedback=data.get("feedback"),
            improvement_suggestions=data.get("improvement_suggestions"),
            encouragement=data.get("encouragement"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )