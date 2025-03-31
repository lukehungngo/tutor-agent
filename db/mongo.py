from pymongo import MongoClient
from bson.objectid import ObjectId
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from models.exam import Question
from models.document_info import DocumentInfo
from config.settings import settings

class MongoDB:
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize MongoDB connection."""
        self.client = MongoClient(connection_string or settings.mongodb_uri)
        self.db = self.client[settings.mongodb_collection_name]

    def save_document_info(self, doc_info: DocumentInfo) -> str:
        """Save document information to MongoDB."""
        # Convert document to dictionary for MongoDB
        doc_dict = {
            "title": doc_info.title,
            "filename": doc_info.filename,
            "session_id": doc_info.session_id,
            "author": doc_info.author,
            "description": doc_info.description,
            "tags": doc_info.tags,
            "file_size": doc_info.file_size,
            "chunk_count": doc_info.chunk_count,
            "created_at": datetime.now(timezone.utc),
        }

        result = self.db.document_info.insert_one(doc_dict)
        return str(result.inserted_id)

    def get_document_info(self, doc_id: str) -> Optional[DocumentInfo]:
        """Get document information by ID."""
        doc_dict = self.db.document_info.find_one({"_id": ObjectId(doc_id)})
        if not doc_dict:
            return None
            
        # Convert ObjectId to string
        doc_dict["_id"] = str(doc_dict["_id"])
        
        return DocumentInfo.from_dict(doc_dict)

    def get_document_info_by_session(self, session_id: str) -> Optional[DocumentInfo]:
        """Get document information by session ID."""
        doc_dict = self.db.document_info.find_one({"session_id": session_id})
        if not doc_dict:
            return None
            
        # Convert ObjectId to string
        doc_dict["_id"] = str(doc_dict["_id"])
        
        return DocumentInfo.from_dict(doc_dict)

    def delete_document_info(self, doc_id: str) -> bool:
        """Delete document information by ID."""
        result = self.db.document_info.delete_one({"_id": ObjectId(doc_id)})
        return result.deleted_count > 0

    def save_question(self, question: Question) -> str:
        """Save a question to the database."""
        question_data = {
            "question": question.question,
            "bloom_level": question.bloom_level.value,
            "hint": question.hint,
            "answer": question.answer,
            "context": question.context,
            "document_id": question.document_id,
            "created_at": datetime.now(timezone.utc),
        }

        result = self.db.questions.insert_one(question_data)
        return str(result.inserted_id)

    def save_questions(self, questions: List[Question]) -> List[str]:
        """Save multiple questions to the database."""
        question_data = []
        for question in questions:
            question_data.append(
                {
                    "question": question.question,
                    "bloom_level": question.bloom_level.value,
                    "hint": question.hint,
                    "answer": question.answer,
                    "context": question.context,
                    "document_id": question.document_id,
                    "created_at": datetime.now(timezone.utc),
                }
            )

        if not question_data:
            return []

        result = self.db.questions.insert_many(question_data)
        return [str(id) for id in result.inserted_ids]

    def get_question(self, question_id: str) -> Optional[Dict]:
        """Get a question by ID."""
        return self.db.questions.find_one({"_id": ObjectId(question_id)})

    def get_questions_by_document(self, document_id: str) -> List[Dict]:
        """Get questions by document ID."""
        return list(self.db.questions.find({"document_id": document_id}))

    def save_answer(
        self,
        document_id: str,
        question_id: str,
        user_id: str,
        answer_text: str,
        score: Optional[float] = None,
        feedback: Optional[str] = None,
    ) -> str:
        """Save a user's answer to a question."""
        answer_data = {
            "document_id": document_id,
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
        """Get a user answer by ID."""
        return self.db.user_answers.find_one({"_id": ObjectId(answer_id)})