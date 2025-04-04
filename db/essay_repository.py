from pymongo import MongoClient
from bson.objectid import ObjectId
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from models.essay import Question, UserAnswer
from models.document_info import DocumentInfo
from config.settings import settings


class EssayRepository:
    def __init__(self):
        """Initialize MongoDB connection."""
        self.client = MongoClient(settings.MONGODB_URI)
        self.db = self.client[settings.MONGODB_DB_NAME]

    def save_document_info(self, doc_info: DocumentInfo) -> str:
        """Save document information to MongoDB."""
        # Convert document to dictionary for MongoDB
        doc_dict = {
            "title": doc_info.title,
            "filename": doc_info.filename,
            "author": doc_info.author,
            "description": doc_info.description,
            "tags": doc_info.tags,
            "file_size": doc_info.file_size,
            "chunk_count": doc_info.chunk_count,
            "created_at": datetime.now(timezone.utc),
        }

        # Add user_id if present
        if doc_info.user_id:
            doc_dict["user_id"] = ObjectId(doc_info.user_id)

        result = self.db.document_info.insert_one(doc_dict)
        return str(result.inserted_id)

    def get_document_info(self, doc_id: str) -> Optional[DocumentInfo]:
        """Get document information by ID."""
        doc_dict = self.db.document_info.find_one({"_id": ObjectId(doc_id)})
        if not doc_dict:
            return None

        # Convert ObjectId to string
        doc_dict["_id"] = str(doc_dict["_id"])
        if "user_id" in doc_dict and isinstance(doc_dict["user_id"], ObjectId):
            doc_dict["user_id"] = str(doc_dict["user_id"])

        return DocumentInfo.from_dict(doc_dict)

    def get_documents_by_user(self, user_id: str) -> List[DocumentInfo]:
        """Get all documents owned by a user.

        Args:
            user_id: User ID to filter documents by

        Returns:
            List of DocumentInfo objects for the user
        """
        documents_data = list(
            self.db.document_info.find({"user_id": ObjectId(user_id)})
        )

        # Convert each document dictionary to a DocumentInfo model
        documents = []
        for doc_data in documents_data:
            # Convert ObjectId to string for serialization
            doc_data["_id"] = str(doc_data.pop("_id"))
            if "user_id" in doc_data and isinstance(doc_data["user_id"], ObjectId):
                doc_data["user_id"] = str(doc_data["user_id"])
            documents.append(DocumentInfo.from_dict(doc_data))

        return documents

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

    def save_questions(
        self,
        questions: List[Question],
    ) -> List[str]:
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
                    "created_at": question.created_at,
                    "user_id": question.user_id,
                }
            )

        if not question_data:
            return []

        result = self.db.questions.insert_many(question_data)
        return [str(id) for id in result.inserted_ids]

    def get_question(self, question_id: str) -> Optional[Question]:
        """Get a question by ID and convert to Question model.

        Args:
            question_id: ID of the question to retrieve

        Returns:
            Question model or None if not found
        """
        question_data = self.db.questions.find_one({"_id": ObjectId(question_id)})
        if not question_data:
            return None

        # Convert ObjectId to string for serialization
        question_data["id"] = str(question_data.pop("_id"))

        # Create Question model from dictionary using from_dict
        return Question.from_dict(question_data)

    def get_questions_by_document(self, document_id: str) -> List[Question]:
        """Get questions by document ID and convert to Question models.

        Args:
            document_id: Document ID to filter questions by

        Returns:
            List of Question models for the document, sorted by ID in descending order
        """
        questions_data = list(
            self.db.questions.find({"document_id": document_id}).sort("_id", -1)
        )

        # Convert each question dictionary to a Question model
        questions = []
        for question_data in questions_data:
            # Convert ObjectId to string for serialization
            question_data["id"] = str(question_data.pop("_id"))
            # Create Question model using from_dict
            questions.append(Question.from_dict(question_data))

        return questions

    def save_answer(self, user_answer: UserAnswer) -> str:
        """Save a user's answer to a question.

        If an answer already exists for the same document, question, and user,
        it will be updated instead of creating a new one.

        Returns:
            The ID of the inserted or updated answer
        """
        # Get answer data as dictionary
        answer_data = user_answer.as_dict()

        # Remove created_at from the main update to prevent conflict
        if "created_at" in answer_data:
            del answer_data["created_at"]

        # Define the filter to find existing answer
        filter_query = {
            "document_id": ObjectId(user_answer.document_id),
            "question_id": ObjectId(user_answer.question_id),
            "user_id": ObjectId(user_answer.user_id),
        }

        # Set created_at only for new documents
        update_data = {
            "$set": answer_data,
            "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
        }

        # Perform upsert operation
        result = self.db.user_answers.update_one(filter_query, update_data, upsert=True)

        # If upserted, get the new ID, otherwise find the existing document to get its ID
        if result.upserted_id:
            return str(result.upserted_id)
        else:
            existing_answer = self.db.user_answers.find_one(filter_query)
            if existing_answer:
                return str(existing_answer["_id"])
            else:
                raise ValueError("No answer found after upsert operation")

    def get_answer_by_id(self, answer_id: str) -> Optional[Dict]:
        """Get a user answer by ID."""
        return self.db.user_answers.find_one({"_id": ObjectId(answer_id)})

    def update_document_info(self, doc_id: str, update_fields: Dict) -> bool:
        """Update document information fields.

        Args:
            doc_id: Document ID to update
            update_fields: Dictionary of fields to update

        Returns:
            True if update was successful, False otherwise
        """
        result = self.db.document_info.update_one(
            {"_id": ObjectId(doc_id)}, {"$set": update_fields}
        )
        return result.modified_count > 0

    def get_user_answers_by_document(
        self, document_id: str, user_id: str
    ) -> List[UserAnswer]:
        """Get all answers submitted by a user for a specific document.

        Args:
            document_id: The ID of the document
            user_id: The ID of the user

        Returns:
            List of user answers for the document
        """
        try:
            answers = self.db.user_answers.find(
                {"document_id": ObjectId(document_id), "user_id": ObjectId(user_id)}
            )

            result = []
            for answer in answers:
                # Convert all ObjectId fields to strings for serialization
                if "_id" in answer:
                    answer["id"] = str(answer.pop("_id"))
                if "user_id" in answer and isinstance(answer["user_id"], ObjectId):
                    answer["user_id"] = str(answer["user_id"])
                if "document_id" in answer and isinstance(
                    answer["document_id"], ObjectId
                ):
                    answer["document_id"] = str(answer["document_id"])
                if "question_id" in answer and isinstance(
                    answer["question_id"], ObjectId
                ):
                    answer["question_id"] = str(answer["question_id"])

                result.append(UserAnswer.from_dict(answer))

            return result
        except Exception as e:
            # Log the error and return empty list instead of propagating the error
            print(f"Error retrieving answers for document {document_id}: {str(e)}")
            return []

    def get_user_answer_by_question_id(
        self, question_id: str, user_id: str
    ) -> Optional[UserAnswer]:
        """Get a user answer by question ID and user ID."""
        answer = self.db.user_answers.find_one(
            {"question_id": ObjectId(question_id), "user_id": ObjectId(user_id)}
        )
        if answer:
            # Convert all ObjectId fields to strings for serialization
            if "_id" in answer:
                answer["id"] = str(answer.pop("_id"))
            if "user_id" in answer and isinstance(answer["user_id"], ObjectId):
                answer["user_id"] = str(answer["user_id"])
            if "document_id" in answer and isinstance(answer["document_id"], ObjectId):
                answer["document_id"] = str(answer["document_id"])
            if "question_id" in answer and isinstance(answer["question_id"], ObjectId):
                answer["question_id"] = str(answer["question_id"])

            return UserAnswer.from_dict(answer)
        return None

    def delete_question(self, question_id: str) -> bool:
        """Delete a question by ID."""
        result = self.db.questions.delete_one({"_id": ObjectId(question_id)})
        return result.deleted_count > 0
