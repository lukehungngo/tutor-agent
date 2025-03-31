# TODO: Move all models of core to this file

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from utils import logger


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


@dataclass
class Question:
    question: str
    bloom_level: BloomLevel
    context: Optional[str] = None
    hint: Optional[str] = None
    answer: Optional[str] = None
    document_id: Optional[str] = None
    
    @staticmethod
    def from_dict(data: Dict) -> "Question":
        return Question(
            question=data["question"],
            bloom_level=BloomLevel(data["bloom_level"]),
            hint=data.get("hint"),
            answer=data.get("answer"),
        )

    def as_dict(self) -> Dict:
        return {
            "question": self.question,
            "bloom_level": self.bloom_level.value,
            "hint": self.hint,
            "answer": self.answer,
            "context": self.context,
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
