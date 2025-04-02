import os

from config import settings
from pydantic import BaseModel


class ResponseSchema(BaseModel):
    answer: str
    confidence: float
    sources: list[str]


open_api_model = settings.OPENAI_CLIENT
google_client = settings.GOOGLE_GEMINI_CLIENT

open_api_structured_llm = open_api_model.with_structured_output(
    ResponseSchema, method="function_calling"
)
google_structured_llm = google_client.with_structured_output(ResponseSchema)

response = open_api_model.invoke("Explain quantum entanglement")
print("Open API response type: ", type(response))
print("Open API response: ", response)

response = google_structured_llm.invoke(response.content)
print("Google response type: ", type(response))
print("Google response: ", response)
