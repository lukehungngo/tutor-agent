from dataclasses import dataclass
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils import time_execution
from pre_trained_model.config import GenerationConfig


@dataclass
class Phi35MiniGenerationConfig(GenerationConfig):
    """Base configuration for text generation."""

    max_new_tokens: int = 1024
    min_new_tokens: int = 150
    temperature: float = 0.1  # Very low for consistent outputs
    top_p: float = 0.3  # More focused sampling
    top_k: int = 20  # Limited token selection
    repetition_penalty: float = 1.3
    length_penalty: float = 0.8  # Favor shorter outputs
    num_beams: int = 4  # Increased beam search
    do_sample: bool = True  # Deterministic generation
    early_stopping: bool = True
    no_repeat_ngram_size: int = 3


class Phi35MiniSummarizer:
    """A summarizer using Microsoft's Phi-3.5-mini-instruct model with improved prompting."""

    SUMMARY_PROMPT = """Below is a text that needs to be summarized. Create a clear, factual summary that:
1. Captures all key points
2. Uses precise, concise language
3. Maintains original meaning
4. Avoids any meta-commentary

Text:
{text}

Write a direct, informative summary:"""

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3.5-mini-instruct",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        default_config: Optional[GenerationConfig] = None,
    ):
        """Initialize the summarizer with model and tokenizer.

        Args:
            model_name: The name of the model to load
            device: Device to use ("cuda", "cpu", "mps", etc.) or None for auto-selection
            dtype: Data type for model weights
            default_config: Default generation configuration
        """
        self.device = device or "mps"
        self.model = self._load_model(model_name, dtype)
        self.tokenizer = self._load_tokenizer(model_name)
        self.config = default_config or Phi35MiniGenerationConfig()

    def _load_model(self, model_name: str, dtype: torch.dtype):
        """Load and configure the model."""
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
            offload_folder="offload",
        )

    def _load_tokenizer(self, model_name: str):
        """Load and configure the tokenizer."""
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def _get_max_length(self):
        """Calculate maximum safe input length for Phi-3.5."""
        # Reserve tokens for generation and account for prompt length
        max_prompt_tokens = 512  # Approximate tokens for the prompt template

        # Use a more reasonable context window size (20K)
        # This balances capability with practical performance
        reasonable_context_window = 200000

        # Calculate max length, ensuring we stay within reasonable limits
        # while reserving space for generation
        return (
            min(reasonable_context_window, self.tokenizer.model_max_length)
            - self.config.max_new_tokens
            - max_prompt_tokens
        )

    def _prepare_input(self, text: str, prompt_template=None):
        """Prepare input for model inference.

        Args:
            text: Input text to summarize

        Returns:
            Tokenized input ready for the model
        """
        # Format with the improved structured prompt
        prompt = prompt_template or self.SUMMARY_PROMPT.format(text=text)

        # Tokenize with appropriate truncation
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._get_max_length(),
        )

        # Move to proper device
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _clean_summary(self, text: str):
        """Clean and format the generated summary.

        Args:
            text: Raw generated text

        Returns:
            Cleaned summary text
        """
        # Extract only the generated summary if our prompt was included in output
        if "Summary:" in text:
            summary = text.split("Summary:")[-1].strip()
        else:
            summary = text.strip()

        # Remove any trailing text after conclusive sentence (e.g. self-commentary)
        for delimiter in ["\n\n", "\n"]:
            parts = summary.split(delimiter, 1)
            if (
                len(parts) > 1 and len(parts[0]) > 50
            ):  # Only split if first part is substantial
                summary = parts[0]
                break

        return summary

    @torch.no_grad()
    @time_execution
    def generate_summary(
        self,
        text: str,
        config: Optional[GenerationConfig] = None,
        custom_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate a summary of the input text.

        Args:
            text: Input text to summarize
            config: Optional generation configuration object
            custom_config: Optional dictionary of generation parameters

        Returns:
            Generated summary text
        """
        try:
            # Prepare inputs
            prompt_template = (
                custom_config["prompt_template"] if custom_config else None
            )
            inputs = self._prepare_input(text, prompt_template=prompt_template)

            # Prepare generation configuration
            raw_config = config.as_dict() if config else self.config.as_dict()

            # Filter out unsupported parameters
            supported_params = [
                "max_new_tokens",
                "min_new_tokens",
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "length_penalty",
                "num_beams",
                "do_sample",
                "early_stopping",
                "no_repeat_ngram_size",
                "penalty_alpha",
            ]

            generation_config = {
                k: v for k, v in raw_config.items() if k in supported_params
            }

            # Add custom config and required tokens
            if custom_config:
                generation_config.update(custom_config)

            generation_config.update(
                {
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
            )

            # Generate summary
            outputs = self.model.generate(**inputs, **generation_config)

            # Process and clean output
            decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = self._clean_summary(decoded_text)

            return summary

        except Exception as e:
            print(f"Error during summary generation: {str(e)}")
            return ""


@time_execution
def main():
    """Example usage of the summarizer."""

    # Initialize summarizer
    summarizer = Phi35MiniSummarizer()

    # Example text
    input_text = """
    Large language models (LLMs) are AI systems trained on vast amounts of text data.
    They can understand and generate human-like text, making them useful for various
    applications like writing, translation, and answering questions.
    """

    # Generate summary with default settings
    summary = summarizer.generate_summary(input_text)
    print("\nGenerated Summary:")
    print(summary)
    # shorter while still capturing all the essential information. It efficiently conveys what LLMs are, what they're trained on, what they can do, and their applications.


if __name__ == "__main__":
    main()
