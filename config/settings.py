from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
import os

os.environ["GRPC_PYTHON_LOG_LEVEL"] = "0"
load_dotenv(".env")
SUPPORTED_MODELS = ["gpt-4o-mini-2024-07-18", "gemini-2.0-flash"]


class Settings:
    openai_api_key: SecretStr = SecretStr(os.getenv("OPENAI_API_KEY", ""))
    open_api_client = ChatOpenAI(model=SUPPORTED_MODELS[0], api_key=openai_api_key)
    google_gemini_api_key: SecretStr = SecretStr(os.getenv("GEMINI_API_KEY", ""))
    llm = open_api_client
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    mongodb_uri = os.getenv("MONGODB_URI")
    mongodb_collection_name = os.getenv("MONGODB_COLLECTION_NAME") or "tutor_agent"
    
    def __init__(self):  # Initialize the gemini client when settings instance is called
        self.google_gemini_client: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
            api_key=self.google_gemini_api_key,
            model=SUPPORTED_MODELS[1],
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=3,
        )


settings = Settings()
