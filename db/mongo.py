from pymongo import MongoClient
from bson.objectid import ObjectId
from typing import List, Dict, Optional
from datetime import datetime, timezone
from models.exam import Question, BloomLevel
from config.settings import settings

class MongoDB:
    def __init__(self, collection_name: str = settings.mongodb_collection_name):
        """Initialize MongoDB connection.

        Args:
            connection_string: MongoDB connection string, if None will use MONGODB_URI environment variable
        """
        self.client = MongoClient(settings.mongodb_uri)
        self.db = self.client[collection_name]

    def save_question(self, question: Question) -> str:
        """Save a question to the database.

        Args:
            question: Question object

        Returns:
            ID of the inserted question
        """
        question_data = {
            "question": question.question,
            "bloom_level": question.bloom_level.value,
            "hint": question.hint,
            "answer": question.answer,
            "context": question.context,
            "created_at": datetime.now(timezone.utc),
        }

        result = self.db.questions.insert_one(question_data)
        return str(result.inserted_id)

    def save_questions(self, questions: List[Question]) -> List[str]:
        """Save multiple questions to the database.

        Args:
            questions: List of Question objects

        Returns:
            List of inserted question IDs
        """
        question_data = []
        for question in questions:
            question_data.append(
                {
                    "question": question.question,
                    "bloom_level": question.bloom_level.value,
                    "hint": question.hint,
                    "answer": question.answer,
                    "context": question.context,
                    "created_at": datetime.now(timezone.utc),
                }
            )

        if not question_data:
            return []

        result = self.db.questions.insert_many(question_data)
        return [str(id) for id in result.inserted_ids]

    def get_question(self, question_id: str) -> Optional[Dict]:
        """Get a question by ID.

        Args:
            question_id: ID of the question

        Returns:
            Question data dictionary or None if not found
        """
        return self.db.questions.find_one({"_id": ObjectId(question_id)})

    def get_questions_by_bloom_level(self, bloom_level: BloomLevel) -> List[Dict]:
        """Get questions by Bloom's Taxonomy level.

        Args:
            bloom_level: BloomLevel enum

        Returns:
            List of question dictionaries
        """
        return list(self.db.questions.find({"bloom_level": bloom_level.value}))

    def save_answer(
        self,
        question_id: str,
        user_id: str,
        answer_text: str,
        score: Optional[float] = None,
        feedback: Optional[str] = None,
    ) -> str:
        """Save a user's answer to a question.

        Args:
            question_id: ID of the question
            user_id: ID of the user
            answer_text: User's answer text
            score: Optional score (0-100)
            feedback: Optional feedback text

        Returns:
            ID of the inserted answer
        """
        answer_data = {
            "question_id": ObjectId(question_id),
            "user_id": user_id,
            "answer_text": answer_text,
            "score": score,
            "feedback": feedback,
            "created_at": datetime.now(timezone.utc),
        }

        result = self.db.user_answers.insert_one(answer_data)
        return str(result.inserted_id)

    def get_answer_by_id(self, answer_id: str) -> Optional[Dict]:
        """Get a user answer by ID.

        Args:
            answer_id: ID of the answer

        Returns:
            Answer data dictionary or None if not found
        """
        return self.db.user_answers.find_one({"_id": ObjectId(answer_id)})
