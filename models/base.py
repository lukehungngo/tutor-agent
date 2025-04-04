from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import json


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
class MultipleChoiceQuestion:
    id: Optional[str] = None
    question: str = ""
    bloom_level: BloomLevel = BloomLevel.REMEMBER
    options: Dict[str, str] = field(default_factory=dict)
    answer: str = ""
    hint: str = ""
    explanation: str = ""

    def as_dict(self) -> Dict:
        """Convert MultipleChoiceQuestion to dictionary."""
        return {
            "id": self.id,
            "bloom_level": (
                self.bloom_level.value
                if isinstance(self.bloom_level, BloomLevel)
                else self.bloom_level
            ),
            "question": self.question,
            "options": self.options,
            "answer": self.answer,
            "hint": self.hint,
            "explanation": self.explanation,
        }

    @staticmethod
    def from_dict(response: Dict) -> "MultipleChoiceQuestion":
        options_data = response.get("options", {})
        if isinstance(options_data, str):
        # If it's a string, parse it as JSON
            options = json.loads(options_data)
        else:
            # If it's already a list, use it directly
            options = options_data
        return MultipleChoiceQuestion(
            id=response.get("id", None),
            question=response.get("question", ""),
            bloom_level=BloomLevel(response.get("bloom_level", BloomLevel.REMEMBER.value)),
            options=options,
            answer=response.get("answer", ""),
            hint=response.get("hint", ""),
            explanation=response.get("explanation", ""),
        )
    
    @staticmethod
    def from_dict_list(response: List[Dict]) -> List["MultipleChoiceQuestion"]:
        return [MultipleChoiceQuestion.from_dict(question) for question in response]

    @staticmethod
    def from_raw_response_list(response: Dict) -> List["MultipleChoiceQuestion"]:
        """Convert a raw LLM response into a list of MultipleChoiceQuestion objects."""
        if not isinstance(response, Dict) or "questions" not in response:
            return []
        questions_dict = response.get("questions", [])
        return [
            MultipleChoiceQuestion.from_dict(question) for question in questions_dict
        ]
