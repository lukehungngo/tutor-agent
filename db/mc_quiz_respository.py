from pymongo import MongoClient
from config.settings import settings
from models import TopicMcQuiz
from typing import Optional, List
from bson.objectid import ObjectId
from utils.logger import logger


class QuizMCRepository:
    def __init__(self):
        """Initialize MongoDB connection."""
        self.client = MongoClient(settings.MONGODB_URI)
        self.db = self.client[settings.MONGODB_DB_NAME]

    def save_mc_quiz(self, mc_quiz: TopicMcQuiz) -> str:
        """Save a quiz MC to the database."""
        try:
            mc_quiz_data = mc_quiz.as_dict()
            
            # Handle ID fields correctly
            if "id" in mc_quiz_data:
                del mc_quiz_data["id"]
            
            # Convert user_id to ObjectId if it's a string
            if "user_id" in mc_quiz_data and isinstance(mc_quiz_data["user_id"], str):
                mc_quiz_data["user_id"] = ObjectId(mc_quiz_data["user_id"])
                
            # We're keeping correct_answers as a JSON string intentionally
            # No conversion to dictionary needed
            
            result = self.db.mc_quizs.insert_one(mc_quiz_data)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error saving quiz: {str(e)}")
            raise

    def get_mc_quiz_by_id(self, mc_quiz_id: str) -> Optional[TopicMcQuiz]:
        """Get a quiz MC by ID."""
        try:
            mc_quiz_data = self.db.mc_quizs.find_one({"_id": ObjectId(mc_quiz_id)})
            if not mc_quiz_data:
                logger.error(f"Quiz MC with ID {mc_quiz_id} not found")
                return None
            # Convert ObjectId fields to strings for serialization
            if "_id" in mc_quiz_data:
                mc_quiz_data["id"] = str(mc_quiz_data.pop("_id"))
            if "user_id" in mc_quiz_data and isinstance(mc_quiz_data["user_id"], ObjectId):
                mc_quiz_data["user_id"] = str(mc_quiz_data["user_id"])
                
            # We're keeping correct_answers as a JSON string
            # The model will handle it correctly
            return TopicMcQuiz.from_dict(mc_quiz_data)
        except Exception as e:
            logger.error(f"Error retrieving quiz {mc_quiz_id}: {str(e)}")
            return None

    def get_mc_quiz_by_user_id(self, user_id: str, exclude_heavy_data: bool = False) -> List[TopicMcQuiz]:
        """Get all quiz MCs by user ID."""
        try:
            # Convert string ID to ObjectId if it's not already
            user_object_id = ObjectId(user_id) if isinstance(user_id, str) else user_id
            
            # Find all quizzes for this user
            mc_quiz_data = list(self.db.mc_quizs.find({"user_id": user_object_id}))
            # Convert MongoDB documents to model objects
            results = []
            for mc_quiz in mc_quiz_data:
                # Convert ObjectId fields to strings for serialization
                if "_id" in mc_quiz:
                    mc_quiz["id"] = str(mc_quiz.pop("_id"))
                if "user_id" in mc_quiz and isinstance(mc_quiz["user_id"], ObjectId):
                    mc_quiz["user_id"] = str(mc_quiz["user_id"])
                
                # We're keeping correct_answers as a JSON string
                # No need to parse it - the model should handle it correctly
                
                # Convert to model object
                quiz_model = TopicMcQuiz.from_dict(mc_quiz, exclude_heavy_data=exclude_heavy_data)
                results.append(quiz_model)
            
            return results
        except Exception as e:
            print(f"Error retrieving quizzes for user {user_id}: {str(e)}")
            # Return empty list instead of raising exception
            return []

    def update_mc_quiz(self, mc_quiz: TopicMcQuiz):
        """Update a quiz MC."""
        try:
            # Get dict representation of quiz
            mc_quiz_data = mc_quiz.as_dict()
            
            # Handle ID conversion - make sure we don't try to update ID
            if "id" in mc_quiz_data:
                del mc_quiz_data["id"]
            
            # Convert user_id to ObjectId if it's a string
            if "user_id" in mc_quiz_data and isinstance(mc_quiz_data["user_id"], str):
                mc_quiz_data["user_id"] = ObjectId(mc_quiz_data["user_id"])
            
            # We're keeping correct_answers as a JSON string intentionally
            # No conversion to dictionary needed
            
            # Update in MongoDB
            result = self.db.mc_quizs.update_one(
                {"_id": ObjectId(mc_quiz.id)}, {"$set": mc_quiz_data}
            )
            
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating quiz {mc_quiz.id}: {str(e)}")
            raise
