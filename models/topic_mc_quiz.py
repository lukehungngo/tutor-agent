from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from models.base import MultipleChoiceQuestion
import json
from utils import logger


@dataclass
class TopicMcQuiz:
    """Document information model for the AI Tutor, distinct from LangChain's Document."""

    id: Optional[str] = None
    user_id: Optional[str] = None
    topic: str = ""
    description: str = ""
    questions: List[MultipleChoiceQuestion] = field(default_factory=list)
    correct_answers: Dict[str, str] = field(default_factory=dict)
    chosen_answers: Dict[str, str] = field(default_factory=dict)
    total_questions: int = 0
    total_score: int = 0
    is_completed: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @staticmethod
    def from_dict(data: Dict, exclude_heavy_data: bool = False) -> "TopicMcQuiz":
        """Convert a dictionary to TopicQuizMC object."""
        try:
            # Handle questions field - could be a list or a JSON string
            if not exclude_heavy_data:
                questions_data = data.get("questions", [])
                if isinstance(questions_data, str):
                    # If it's a string, parse it as JSON
                    questions = MultipleChoiceQuestion.from_dict_list(json.loads(questions_data))
                else:
                    # If it's already a list, use it directly
                    questions = MultipleChoiceQuestion.from_dict_list(questions_data)
                correct_answers = data.get("correct_answers", {})
                if not correct_answers:
                    correct_answers = {}
                chosen_answers = data.get("chosen_answers", {})
                if not chosen_answers:
                    chosen_answers = {}
            else:
                questions = []
                correct_answers = {}
                chosen_answers = {}
            # Handle datetime fields safely
            created_at = None
            updated_at = None
            
            if data.get("created_at"):
                try:
                    if isinstance(data["created_at"], datetime):
                        created_at = data["created_at"]
                    elif isinstance(data["created_at"], str) and data["created_at"].strip():
                        created_at = datetime.fromisoformat(data["created_at"])
                except Exception as e:
                    print(f"Error parsing created_at date: {data.get('created_at')} - {str(e)}")
            
            if data.get("updated_at"):
                try:
                    if isinstance(data["updated_at"], datetime):
                        updated_at = data["updated_at"]
                    elif isinstance(data["updated_at"], str) and data["updated_at"].strip():
                        updated_at = datetime.fromisoformat(data["updated_at"])
                except Exception as e:
                    print(f"Error parsing updated_at date: {data.get('updated_at')} - {str(e)}")
                
            return TopicMcQuiz(
                id=data.get("_id") or data.get("id"),
                user_id=data.get("user_id", None),
                topic=data.get("topic", ""),
                description=data.get("description", ""),
                questions=questions,
                total_questions=data.get("total_questions", 0),
                correct_answers=correct_answers,
                chosen_answers=chosen_answers,
                total_score=data.get("total_score", 0),
                is_completed=data.get("is_completed", False),
                created_at=created_at,
                updated_at=updated_at,
            )
        
        except Exception as e:
            print(f"Error in TopicMcQuiz.from_dict: {str(e)}")
            # Return an empty quiz object instead of raising exception
            return TopicMcQuiz()

    def as_dict(self) -> Dict:
        """Convert DocumentInfo to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "topic": self.topic,
            "description": self.description,
            "total_questions": self.total_questions,
            "questions": [question.as_dict() for question in self.questions],
            "correct_answers": self.correct_answers,
            "chosen_answers": self.chosen_answers,
            "total_score": self.total_score,
            "is_completed": self.is_completed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @staticmethod
    def from_raw_response(response: Dict) -> "TopicMcQuiz":
        """Convert a raw LLM response into a TopicQuizMC object.

        Args:
            response: Raw dictionary response from LLM containing questions
            context: The context to associate with the questions

        Returns:
            TopicQuizMC object
        """
        topic_mc_quiz = TopicMcQuiz()
        if not isinstance(response, Dict) or "questions" not in response:
            return topic_mc_quiz

        for raw in response["questions"]:
            try:
                question = MultipleChoiceQuestion.from_dict(raw)
                topic_mc_quiz.questions.append(question)
            except (KeyError, ValueError) as e:
                logger.error(f"Error processing question: {e}")
                continue

        return topic_mc_quiz
