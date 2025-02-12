import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

class Settings:
    openai_api_key : SecretStr = SecretStr(os.getenv("OPENAI_API_KEY", ""))
    open_api_model = "gpt-3.5-turbo"
    open_api_client = ChatOpenAI(model=open_api_model, api_key=openai_api_key)
    tavily_api_key = os.getenv("TAVILY_API_KEY")

settings = Settings()
