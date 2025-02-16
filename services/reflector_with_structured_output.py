from pydantic import BaseModel
from models.state import ResearchState, ResearchResult
from langgraph.types import Command
from typing import Any

class StructuredOutputReflector:
    """Agent that processes research results into structured output format."""
    
    structured_llm: Any  # Replace with actual LLM type
    structured_output_type: Any
    def __init__(self, llm: Any):
        self.structured_output_type = ResearchResult
        self.structured_llm = llm.with_structured_output(self.structured_output_type)
        self.llm = llm
    
    def run(self, state: ResearchState) -> Command:
        """Main entry point for the reflector agent."""
        try:
            messages = state.get("messages", [])
            last_message_content = messages[-1].content if messages else ""
            
            # First, generate a summary insight
            summary_prompt = """
                Summarize the following text in one clear and concise paragraph, capturing the key ideas without missing critical points. Ensure the summary is easy to understand and avoids excessive detail  
                Content to analyze: {content}
                Output format should be string only, no markdown or other formatting.
            """.format(content=last_message_content)
            summary = self.llm.invoke(summary_prompt)
            
            # Then get the structured response with key fields
            structured_prompt = """
                Analyze the following content and extract key information into a structured format.
                Keep the original titles, URLs and contents if present, but organize the information clearly.
                Focus on maintaining source attribution while presenting the content in an accessible way.

                Content to analyze: {content}
                First, you will sort output based on difficulty level based on the summary.
                Second, you will also sort output based on relevance and confidence score to the summary.
                
                Ensure the response includes:
                - Title (preserve original)
                - URL (preserve original if present) 
                - Content summary (preserve original if present)
                - Difficulty level (if applicable)
                - Author/Source (if available)
            """.format(content=last_message_content)
            structured_response = self.structured_llm.invoke(structured_prompt)
            
            # Update the summary field
            structured_response.summary = summary.content
            
            return Command(
                update={
                    "structured_output": structured_response,
                    "messages": [{"role": "assistant", "content": str(structured_response.model_dump())}],
                },
            )
        except Exception as e:
            print("Error: ", e)
            return Command(
                update={
                    "structured_output": None,
                    "messages": [{"role": "assistant", "content": "Failed to structure response."}],
                },
            )