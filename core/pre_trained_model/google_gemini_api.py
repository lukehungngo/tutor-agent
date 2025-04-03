from typing import Dict, Optional, Union, Any
import json
import re
from utils import logger, time_execution
from config.settings import settings


class GoogleGeminiAPI:
    """Implementation for Google's Gemini API with robust JSON support using LangChain."""

    def __init__(
        self,
        temperature: float = 0.1,
        max_tokens: int = 65356,
    ):
        """Initialize the Gemini API client.

        Args:
            temperature: Temperature for text generation (lower is more deterministic)
            max_tokens: Maximum number of tokens to generate
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = settings.GOOGLE_GEMINI_CLIENT

        # Update client parameters with our defaults
        self.client.temperature = temperature
        # Note: max_tokens is handled during individual requests

    def _extract_json_string(self, text: str) -> str:
        """Extract a JSON string from text using regex pattern matching."""
        # Find text that looks like a JSON object (between curly braces)
        pattern = r"({[\s\S]*})"
        matches = re.findall(pattern, text)

        if matches:
            # Return the longest match (most likely the complete JSON)
            return max(matches, key=len)
        return text

    def _format_prompt_for_json(self, prompt: str, schema: Dict) -> str:
        """Format a prompt to encourage proper JSON output."""
        # Convert schema to a string representation
        schema_str = json.dumps(schema, indent=2)

        # Add explicit formatting instructions
        formatted_prompt = (
            f"{prompt}\n\n"
            f"RESPONSE FORMAT:\n{schema_str}\n\n"
            f"IMPORTANT GUIDELINES:\n"
            f"1. Your entire response must be valid JSON starting with '{{' and ending with '}}'.\n"
            f"2. Do not include any text before or after the JSON object.\n"
            f"3. When evaluating answers, be critical and objective:\n"
            f"   - For nonsensical text, gibberish or random characters, ALWAYS use 'incorrect' with score 0\n"
            f"   - Only mark as 'correct' if the answer demonstrates clear understanding\n"
            f"   - Use 'partially_correct' for answers with some valid content but significant gaps\n"
            f"4. Match the schema exactly - use the keys and value types shown in the example.\n"
        )

        return formatted_prompt

    def cleanup(self):
        """Explicitly clean up resources to prevent memory and semaphore leaks."""
        pass

    @time_execution
    def generate(
        self,
        prompt: str,
        schema: Optional[Dict] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Union[Dict, str]:
        """Generate and parse a JSON response from the model.

        Args:
            prompt: The prompt to send to the model
            schema: Optional schema example to guide the output format
            **kwargs: Additional arguments to pass to the generation

        Returns:
            Parsed JSON as a dictionary or raw text
        """
        # Use default schema if none provided
        if schema is None:
            return self.generate_text(prompt, **kwargs)

        # Format the prompt for JSON generation
        formatted_prompt = self._format_prompt_for_json(prompt, schema)

        # Set generation parameters optimized for JSON
        temperature = self.temperature if temperature is None else temperature
        json_kwargs = {
            "temperature": min(0.3, temperature),  # Lower temperature for JSON
            **kwargs,
        }

        # Get raw text response
        raw_response = self.generate_text(formatted_prompt, **json_kwargs)

        # Try to parse JSON from the response
        try:
            # First try to extract a JSON string
            json_str = self._extract_json_string(raw_response)

            # Parse the JSON string
            data = json.loads(json_str)
            return data

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            # Return error information
            return {
                "error": "Failed to parse JSON response",
                "raw_text": raw_response,
                "details": str(e),
            }

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using the Gemini API.

        Args:
            prompt: The prompt to generate text from
            **kwargs: Additional parameters for text generation

        Returns:
            Generated text
        """
        try:
            # Temporarily update client parameters if needed
            original_temp = self.client.temperature

            if "temperature" in kwargs:
                self.client.temperature = kwargs.pop("temperature")

            # Pass max_tokens to the invoke method if specified (with correct parameter name)
            additional_params = {}

            # Generate the response
            response = self.client.invoke(prompt, **additional_params)

            # Extract content as string
            if hasattr(response, "content"):
                if isinstance(response.content, str):
                    result = response.content
                else:
                    # Handle case where content might be a list or another structure
                    result = str(response.content)
            else:
                # Fallback if content is not available
                result = str(response)

            # Restore original parameters
            self.client.temperature = original_temp

            return result

        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            raise RuntimeError(f"Error during text generation: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Create and test the API client
    logger.info("Initializing Gemini API client...")
    try:
        # Initialize the client
        gemini = GoogleGeminiAPI(
            temperature=0.1,  # Very low temperature for predictable JSON
        )

        # Define a simple prompt template
        template = """
        Generate questions about the following context:
        
        CONTEXT: {context}
        
        Create {number_of_questions} questions for "analyze" level based on Bloom's Taxonomy.
        Create {number_of_questions} questions for "create" level based on Bloom's Taxonomy.
        
        Guidelines:
        - analyze: Test analysis of concepts
        - create: Test creation of something new or alternative solutions
        """

        # Format the prompt
        prompt = template.format(
            context="Supervised learning vs unsupervised learning in machine learning",
            number_of_questions=2,
        )

        # Generate JSON response
        logger.info("Generating JSON response...")
        json_response = gemini.generate(
            prompt,
            schema={
                "questions": [
                    {
                        "level": "analyze",
                        "question": "What is supervised learning?",
                        "hint": "Think about labeled data",
                        "answer": "Supervised learning is a type of machine learning where models learn from labeled training data.",
                    }
                ]
            },
        )
        logger.info(f"JSON response: {json.dumps(json_response, indent=2)}")

        # Generate text response
        logger.info("Generating text response...")
        text_response = gemini.generate_text(prompt)
        logger.info(f"Text response: {text_response}")

    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"\nAn error occurred: {e}", exc_info=True)
