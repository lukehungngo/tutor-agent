from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
import os
from pathlib import Path

os.environ["GRPC_PYTHON_LOG_LEVEL"] = "0"
load_dotenv(".env")
SUPPORTED_MODELS = ["gpt-4o-mini-2024-07-18", "gemini-2.0-flash"]

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class Settings:
    OPENAI_API_KEY: SecretStr = SecretStr(os.getenv("OPENAI_API_KEY", ""))
    OPENAI_CLIENT = ChatOpenAI(model=SUPPORTED_MODELS[0], api_key=OPENAI_API_KEY)
    GOOGLE_GEMINI_API_KEY: SecretStr = SecretStr(os.getenv("GEMINI_API_KEY", ""))
    DEFAULT_LLM = OPENAI_CLIENT
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    MONGODB_URI = os.getenv("MONGODB_URI")
    MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME") or "tutor_agent"
    CHROMA_PERSIST_DIR = os.getenv(
        "CHROMA_PERSIST_DIR", str(PROJECT_ROOT / ".chroma_db")
    )
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
    JWT_SECRET_KEY = "698ac142d9aff24ec8d18265c58089c1f0fa68304f28375242e58b75aa2ddde8"
    JWT_EXPIRATION_TIME_MINUTES = 60 * 24 * 7  # 7 days
    JWT_ALGORITHM = "HS256"

    def __init__(self):  # Initialize the gemini client when settings instance is called
        self.GOOGLE_GEMINI_CLIENT: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
            api_key=self.GOOGLE_GEMINI_API_KEY,
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=3,
        )


settings = Settings()
