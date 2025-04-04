from .user_repository import UserRepository
from .essay_repository import EssayRepository
from config.settings import settings
from pymongo import MongoClient
from .chroma_embedding import ChromaEmbeddingStore
from .mc_quiz_respository import QuizMCRepository

mongo_db_client = MongoClient(settings.MONGODB_URI)
mongo_db = mongo_db_client[settings.MONGODB_DB_NAME]

__all__ = [
    "UserRepository",
    "EssayRepository",
    "mongo_db",
    "mongo_db_client",
    "ChromaEmbeddingStore",
    "QuizMCRepository",
]
