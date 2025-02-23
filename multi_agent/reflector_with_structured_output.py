from models.state import State
from langgraph.types import Command
from typing import Any

SUMMARY_PROMPT = """Given the following content, provide a comprehensive analysis and summary:

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
A natural, flowing markdown document that reads like an expert analysis. DO NOT use numbered sections or the internal structure headers above.

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

[Continue with your analysis in this natural style, avoiding any numbered sections or internal structure headers...]
"""


class StructuredOutputReflector:
    """Agent that processes research results into structured output format."""

    def __init__(self, llm: Any, google_gemini_llm: Any):
        self.llm = llm
        self.google_gemini_llm = google_gemini_llm

    def run(self, state: State) -> Command:
        """
        Main entry point for the reflector agent.

        Args:
            state: Current state containing messages and research results

        Returns:
            Command with updated state including structured analysis
        """
        try:
            messages = state.get("messages", [])
            last_message_content = messages[-1].content if messages else ""

            # Generate summary using structured prompt
            summary = self.google_gemini_llm.invoke(
                SUMMARY_PROMPT.format(content=last_message_content)
            )

            # Clean up any remaining structure headers
            cleaned_summary = self._clean_structure_headers(
                summary.content if hasattr(summary, "content") else str(summary)
            )

            return Command(
                update={
                    "summary": cleaned_summary,
                    "messages": [
                        {
                            "role": "assistant",
                            "content": last_message_content,
                        }
                    ],
                },
            )

        except Exception as e:
            print("Error during reflection:", e)
            return Command(
                update={
                    "structured_output": None,
                    "messages": [
                        {
                            "role": "assistant",
                            "content": f"Failed to structure response: {str(e)}",
                        }
                    ],
                },
            )

    def _clean_structure_headers(self, response: str) -> str:
        """Remove any internal structure headers that might have leaked into the output."""
        import re

        # List of patterns to remove
        patterns = [
            r"^Overview and Context\n",
            r"^Key Concepts and Definitions\n",
            r"^Technical Analysis\n",
            r"^Impact and Implications\n",
            r"^Conclusion\n",
            r"^\d+\.\s+(Overview|Key Concepts|Technical|Impact|Conclusion)",
            r"^INTERNAL STRUCTURE:",
            r"^YOUR RESPONSE SHOULD BE:",
            r"^FORMATTING GUIDELINES:",
        ]

        cleaned = response
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE)

        # Remove multiple consecutive newlines
        cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
        return cleaned.strip()
