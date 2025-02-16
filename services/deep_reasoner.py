from typing import Any
from models.state import State, ReasoningResult
from langgraph.types import Command

REASONING_PROMPT = """Given the following content, provide a clear and insightful analysis:

CONTENT TO ANALYZE:
{content}

Follow this internal analysis process (do not include these steps or any numbered sections in output):
- Identify key points and main message
- Extract critical details and relationships
- Analyze implications and insights
- Evaluate technical aspects
- Consider applications and impact

INTERNAL STRUCTURE (IMPORTANT: DO NOT INCLUDE THESE HEADERS OR NUMBERS IN YOUR OUTPUT):
[These headers are for your internal organization only. Your response should flow naturally without these explicit sections]
1. Overview and Context
2. Key Concepts and Definitions
3. Technical Analysis
4. Impact and Implications
5. Conclusion

YOUR RESPONSE SHOULD BE:
A natural, flowing markdown document that reads like an expert analysis. DO NOT use numbered sections or the internal structure headers above. Instead:

FORMATTING GUIDELINES:
- Start with a natural introduction
- Use descriptive headers that fit your specific analysis (not the internal structure headers)
- Use ## for section headers
- Use ### for subtopics (sparingly)
- Use *italics* for emphasis and important terms
- Use `code` for technical terms and implementations
- Use > for impactful quotes or key insights
- Use bullet points for:
  - Related items
  - Feature lists
  - Key implications
- Use --- for natural topic transitions (when needed)
- Use collapsible details for technical deep-dives:
  ```markdown
  <details>
  <summary>Technical Implementation</summary>
  Detailed technical content here...
  </details>
  ```

Keep the total length moderate (around 300-500 words). Focus on insights and implications. The document should flow naturally while using markdown elements to enhance understanding.

EXAMPLE STYLE:
The emergence of *large language models* has fundamentally transformed our understanding of artificial intelligence and its capabilities.

### Recent Developments
The integration of `transformer architectures` into modern AI systems has led to unprecedented advances in natural language understanding and generation.

> "The most significant breakthrough lies in the system's ability to learn and adapt from diverse data sources"

Key implications for the field include:
- Enhanced language understanding
- Improved contextual awareness
- More natural human-AI interaction

---

<details>
<summary>Architecture Details</summary>
The system leverages:
- Multi-head attention mechanisms
- Layer normalization
- Residual connections
</details>

[Continue with your analysis in this natural style, avoiding any numbered sections or internal structure headers...]"""

class DeepReasoner:
    """Agent that performs structured reasoning and determines research needs."""

    def __init__(self, llm: Any, google_gemini_llm: Any):
        """
        Initialize the Reasoner with LLMs.
        
        Args:
            llm: The language model to use for structured output
            google_gemini_llm: The Gemini model for enhanced reasoning
        """
        self.structured_output_type = ReasoningResult
        self.structured_llm = llm.with_structured_output(self.structured_output_type)
        self.llm = llm
        self.google_gemini_llm = google_gemini_llm

    def run(self, state: State) -> Command:
        """
        Main entry point for the reasoner agent.
        
        Args:
            state: Current state containing messages and context
            
        Returns:
            Command with updated state including reasoning results
        """
        try:
            messages = state.get("messages", [])
            last_message = messages[-1].content if messages else ""
            
            # Generate reasoning using Gemini for better analysis
            reasoning_response = self.llm.invoke(
                REASONING_PROMPT.format(content=last_message)
            )
            reasoning_content = reasoning_response.content
            
            # Clean up any remaining structure headers
            reasoning_content = self._clean_structure_headers(reasoning_content)
            
            # Extract confidence and research need
            confidence_level = self._extract_confidence(reasoning_content)
            needs_research = self._assess_research_need(reasoning_content)
            
            # Create structured reasoning result
            structured_response = ReasoningResult(
                question=last_message,
                reasoning_process=reasoning_content,
                confidence_level=confidence_level,
                used_context=False
            )

            return Command(
                update={
                    "structured_output": structured_response,
                    "messages": [
                        {
                            "role": "assistant",
                            "content": str(structured_response.model_dump()),
                        }
                    ],
                },
            )

        except Exception as e:
            print("Error during reasoning:", e)
            return Command(
                update={
                    "structured_output": None,
                    "messages": [
                        {
                            "role": "assistant",
                            "content": f"Failed to perform reasoning: {str(e)}",
                        }
                    ],
                },
            )

    def _clean_structure_headers(self, response: str) -> str:
        """Remove any internal structure headers that might have leaked into the output."""
        import re
        
        # List of patterns to remove
        patterns = [
            r'^Overview and Context\n',
            r'^Key Concepts and Definitions\n',
            r'^Technical Analysis\n',
            r'^Impact and Implications\n',
            r'^Conclusion\n',
            r'^\d+\.\s+(Overview|Key Concepts|Technical|Impact|Conclusion)',
            r'^INTERNAL STRUCTURE:',
            r'^YOUR RESPONSE SHOULD BE:',
            r'^FORMATTING GUIDELINES:'
        ]
        
        cleaned = response
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
        
        # Remove multiple consecutive newlines
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        return cleaned.strip()

    def _extract_confidence(self, response: str) -> int:
        """Extract confidence level from reasoning response."""
        try:
            if "Confidence Level" in response:
                confidence_line = [line for line in response.split('\n') if "Confidence Level" in line][0]
                confidence = int(''.join(filter(str.isdigit, confidence_line)))
                return min(max(confidence, 1), 10)
            return 5
        except:
            return 5

    def _assess_research_need(self, response: str) -> bool:
        """Assess if the reasoning indicates need for additional research."""
        research_indicators = [
            "would need more information",
            "requires external data",
            "cannot be certain without",
            "would need research",
            "external research would",
            "insufficient information",
            "more data needed",
            "would need verification"
        ]
        
        confidence = self._extract_confidence(response)
        needs_research = (
            confidence < 7 or
            any(indicator.lower() in response.lower() for indicator in research_indicators)
        )
        
        return needs_research
