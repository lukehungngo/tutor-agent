from config import settings
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# This work for model without native structured output support
class ResponseSchema(BaseModel):
    answer: str
    confidence: float
    sources: list[str]


model = settings.DEFAULT_LLM
parser = PydanticOutputParser(pydantic_object=ResponseSchema)
prompt = ChatPromptTemplate.from_template(
    "Answer the question.\n{format_instructions}\nQuestion: {question}"
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser
response = chain.invoke({"question": "Explain blockchain technology"})

print("Response type: ", type(response))
print("Response content: ", response)
