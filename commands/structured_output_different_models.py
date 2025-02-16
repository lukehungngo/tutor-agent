import os

from config import settings
from pydantic import BaseModel

class ResponseSchema(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

open_api_model = settings.open_api_client
google_client = settings.google_gemini_client

open_api_structured_llm = open_api_model.with_structured_output(
    ResponseSchema, method="function_calling"
)
google_structured_llm = google_client.with_structured_output(
    ResponseSchema
)

response = open_api_model.invoke("Explain quantum entanglement")
print("Open API response type: ", type(response))
print("Open API response: ", response)

response = google_structured_llm.invoke(response.content)
print("Google response type: ", type(response))
print("Google response: ", response)