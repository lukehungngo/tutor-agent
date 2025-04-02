from typing import Optional
from pymongo import MongoClient
from config.settings import settings
from models.user import User


class UserRepository:
    def __init__(self):
        """Initialize MongoDB connection."""
        self.client = MongoClient(settings.MONGODB_URI)
        self.db = self.client[settings.MONGODB_DB_NAME]

    def get_user(self, username: str) -> Optional[User]:
        """Get a user by username."""
        user_dict = self.db.users.find_one({"username": username})
        if not user_dict:
            return None

        # Convert _id to id for Pydantic model
        if "_id" in user_dict and "id" not in user_dict:
            user_dict["id"] = str(user_dict["_id"])

        return User(**user_dict)
