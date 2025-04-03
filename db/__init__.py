from .user_repository import UserRepository
from .exam_repository import ExamRepository
from config.settings import settings
from pymongo import MongoClient
from .chroma_embedding import ChromaEmbeddingStore

mongo_db_client = MongoClient(settings.MONGODB_URI)
mongo_db = mongo_db_client[settings.MONGODB_DB_NAME]

__all__ = [
    "UserRepository",
    "ExamRepository",
    "mongo_db",
    "mongo_db_client",
    "ChromaEmbeddingStore",
]
