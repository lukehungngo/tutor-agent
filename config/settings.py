import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

load_dotenv()
SUPPORTED_MODELS = ["gpt-4o-mini-2024-07-18", "gemini-2.0-flash"]

class Settings:
    openai_api_key : SecretStr = SecretStr(os.getenv("OPENAI_API_KEY", ""))
    open_api_client = ChatOpenAI(model=SUPPORTED_MODELS[0], api_key=openai_api_key)
    google_gemini_api_key = os.getenv("GEMINI_API_KEY")
    google_gemini_client = None
    llm = open_api_client
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    def get_gemini_client(self):
        if self.google_gemini_client is None:
            self.google_gemini_client = ChatGoogleGenerativeAI(
                model=SUPPORTED_MODELS[1],
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=3,
            )
        return self.google_gemini_client

settings = Settings()