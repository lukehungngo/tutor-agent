import torch
from typing import Any, List, Dict, Optional, Union
from utils import logger, time_execution
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json
import re


class Gemma3Model:
    """Standalone implementation for Google's Gemma 3 model with robust JSON support."""

    def __init__(
        self,
        model_name: str = "google/gemma-3-1b-it",
        temperature: float = 0.1,
        max_tokens: int = 65356,
        device: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
    ):
        """Initialize the Gemma 3 model.

        Args:
            model_name: Name of the model to load
            temperature: Temperature for text generation (lower is more deterministic)
            max_tokens: Maximum number of tokens to generate
            device: Device to use (cuda, mps, cpu)
            torch_dtype: Torch data type to use
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device
        self.torch_dtype = torch_dtype

        # Set up the model
        self._setup_model()

    def _setup_model(self):
        """Set up the model with proper error handling."""
        try:
            logger.info(f"Loading {self.model_name}...")

            # Determine the appropriate device and dtype if not explicitly provided
            if not self.device:
                if torch.cuda.is_available():
                    self.device = "cuda"
                    if not self.torch_dtype:
                        self.torch_dtype = torch.float16  # Use half precision on GPU
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    self.device = "mps"  # For Apple Silicon
                    if not self.torch_dtype:
                        self.torch_dtype = torch.float16
                else:
                    self.device = "cpu"
                    if not self.torch_dtype:
                        self.torch_dtype = torch.float32  # Use full precision on CPU

            logger.info(
                f"Model {self.model_name} using device: {self.device}, dtype: {self.torch_dtype}"
            )

            # First load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )

            # Then load model with proper device and dtype configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                trust_remote_code=True,
            )

            # Create the pipeline
            self.text_generation = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=self.device,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
            )

            logger.info(f"Successfully loaded {self.model_name}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ValueError(
                f"Failed to load {self.model_name}. Make sure you have accepted the model license on Hugging Face and have the latest transformers library installed."
            )

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
            f"IMPORTANT: Your entire response must be valid JSON starting with '{{' and ending with '}}'. "
            f"Do not include any text before or after the JSON object."
        )

        return formatted_prompt

    @time_execution
    def generate_json(
        self, prompt: str, schema: Optional[Dict] = None, **kwargs
    ) -> Dict:
        """Generate and parse a JSON response from the model.

        Args:
            prompt: The prompt to send to the model
            schema: Optional schema example to guide the output format
            **kwargs: Additional arguments to pass to the generation

        Returns:
            Parsed JSON as a dictionary
        """
        # Use default schema if none provided
        if schema is None:
            schema = {
                "questions": [
                    {
                        "level": "remember",
                        "question": "What is supervised learning?",
                        "hint": "Think about labeled data",
                        "answer": "Supervised learning is a type of machine learning where models learn from labeled training data.",
                    }
                ]
            }

        # Format the prompt for JSON generation
        formatted_prompt = self._format_prompt_for_json(prompt, schema)

        # Set generation parameters optimized for JSON
        json_kwargs = {
            "temperature": min(0.2, self.temperature),  # Lower temperature for JSON
            "max_new_tokens": self.max_tokens,
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

    @time_execution
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using the Gemma 3 model.

        Args:
            prompt: The prompt to generate text from
            **kwargs: Additional parameters for text generation

        Returns:
            Generated text
        """
        try:
            # Format the prompt appropriately for the model
            if "it" in self.model_name:
                # Format for instruction-tuned models
                formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            else:
                # For base models, use the prompt as-is
                formatted_prompt = prompt

            # Set up generation parameters
            generation_kwargs = {
                "temperature": self.temperature,
                "do_sample": self.temperature > 0,
                "max_new_tokens": self.max_tokens,
            }

            # Update with any additional kwargs
            generation_kwargs.update(kwargs)

            # Generate text
            response = self.text_generation(formatted_prompt, **generation_kwargs)[0][
                "generated_text"
            ]

            # Extract the model's response
            if "it" in self.model_name and "<start_of_turn>model" in response:
                # Extract just the model's portion of the response
                result = response.split("<start_of_turn>model")[-1].strip()

                # Remove trailing content if needed
                if "<end_of_turn>" in result:
                    result = result.split("<end_of_turn>")[0].strip()
            else:
                # For base models, just return everything after the prompt
                result = response[len(formatted_prompt) :].strip()

            return result

        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            raise RuntimeError(f"Error during text generation: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Create and test the model
    logger.info("Initializing Gemma 3 model...")
    try:
        # Initialize the model
        gemma = Gemma3Model(
            model_name="google/gemma-3-1b-it",
            device="mps",
            torch_dtype=torch.float16,
            temperature=0.1,  # Very low temperature for predictable JSON
        )

        # Define a simple prompt template
        template = """
        Generate questions about the following topic:
        
        TOPIC: {context}
        
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
        json_response = gemma.generate_json(prompt)

        # Print the response
        logger.info("\nJSON Response:")
        logger.info(json.dumps(json_response, indent=2))

    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"\nAn error occurred: {e}", exc_info=True)
