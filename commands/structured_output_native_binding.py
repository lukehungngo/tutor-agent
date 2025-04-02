from config import settings
from pydantic import BaseModel


class ResponseSchema(BaseModel):
    answer: str
    confidence: float
    sources: list[str]


model = settings.DEFAULT_LLM
structured_llm = model.with_structured_output(ResponseSchema)

response = structured_llm.invoke("Explain quantum entanglement")
print(type(response))
print(response)
