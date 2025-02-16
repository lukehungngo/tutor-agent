from config import settings
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Flexible output will return dict that contain defined response schema, ResponseSchema is supported by langchain
response_schemas = [
    ResponseSchema(name="summary", description="Concise answer summary"),
    ResponseSchema(name="details", description="Technical implementation details"),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
model = settings.llm
prompt = ChatPromptTemplate.from_template(
    "Answer the question.\n{format_instructions}\nQuestion: {question}"
).partial(format_instructions=output_parser.get_format_instructions())

chain = prompt | model | output_parser
response = chain.invoke({"question": "Explain blockchain technology"})

print("Response type: ", type(response))
print("Response content: ", response)
