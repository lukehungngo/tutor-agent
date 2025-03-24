from dataclasses import dataclass
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


@dataclass
class GenerationConfig:
    """Base configuration for text generation."""
    max_new_tokens: int = 150
    min_new_tokens: int = 50
    temperature: float = 0.4
    top_p: float = 0.92
    top_k: int = 40
    repetition_penalty: float = 1.25
    length_penalty: float = 1.1
    num_beams: int = 2
    do_sample: bool = True
    early_stopping: bool = True
    no_repeat_ngram_size: int = 3
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary for model generation."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class SummaryGenerationConfig(GenerationConfig):
    """Configuration optimized for balanced summary generation."""
    pass


@dataclass
class ConciseGenerationConfig(GenerationConfig):
    """Configuration optimized for concise, focused output."""
    min_new_tokens: int = 30
    temperature: float = 0.3
    top_p: float = 0.85
    top_k: int = 50
    repetition_penalty: float = 1.5
    length_penalty: float = 1.0
    num_beams: int = 4
    do_sample: bool = True


@dataclass
class CreativeGenerationConfig(GenerationConfig):
    """Configuration optimized for imaginative text generation."""
    max_new_tokens: int = 200
    min_new_tokens: int = 30
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    num_beams: int = 1
    early_stopping: bool = False
    penalty_alpha: float = 0.6


class DeepSeekSummarizer:
    """A summarizer using DeepSeek's distilled LLaMA model with improved prompting."""
    
    SUMMARY_PROMPT = """Provide a comprehensive yet concise summary of the following content:

```
{text}
```

Your summary should:
- Capture all key information, main arguments, and essential points
- Maintain clarity and conciseness (aim for 20-30% of the original length)
- Preserve the original meaning and intent without distortion
- Include all critical details, facts, and conclusions
- Organize information logically, maintaining the relationship between concepts
- Use objective language that reflects the original tone

Focus on extracting what matters most while eliminating redundancy.

Summary:"""
    # microsoft/Phi-3.5-mini-instruct
    # deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    # microsoft/Phi-4-mini-instruct
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3.5-mini-instruct",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        default_config: Optional[GenerationConfig] = None
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
        self.config = default_config or SummaryGenerationConfig()
    
    def _load_model(self, model_name: str, dtype: torch.dtype):
        """Load and configure the model."""
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
            offload_folder="offload"
        )
    
    def _load_tokenizer(self, model_name: str):
        """Load and configure the tokenizer."""
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
    
    def _get_max_length(self):
        """Calculate maximum safe input length."""
        # Reserve tokens for generation and account for prompt length
        max_prompt_tokens = 512  # Approximate tokens for the prompt template
        return min(4096, self.tokenizer.model_max_length) - self.config.max_new_tokens - max_prompt_tokens
    
    def _prepare_input(self, text: str):
        """Prepare input for model inference.
        
        Args:
            text: Input text to summarize
            
        Returns:
            Tokenized input ready for the model
        """
        # Format with the improved structured prompt
        prompt = self.SUMMARY_PROMPT.format(text=text)
        
        # Tokenize with appropriate truncation
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._get_max_length()
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
            if len(parts) > 1 and len(parts[0]) > 50:  # Only split if first part is substantial
                summary = parts[0]
                break
        
        return summary
    
    @torch.no_grad()
    def generate_summary(
        self,
        text: str,
        config: Optional[GenerationConfig] = None,
        custom_config: Optional[Dict[str, Any]] = None
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
            inputs = self._prepare_input(text)
            
            # Prepare generation configuration
            raw_config = config.as_dict() if config else self.config.as_dict()
            
            # Filter out unsupported parameters
            supported_params = [
                "max_new_tokens", "min_new_tokens", "temperature", "top_p", "top_k",
                "repetition_penalty", "length_penalty", "num_beams", "do_sample",
                "early_stopping", "no_repeat_ngram_size", "penalty_alpha"
            ]
            
            generation_config = {
                k: v for k, v in raw_config.items() if k in supported_params
            }
            
            # Add custom config and required tokens
            if custom_config:
                generation_config.update(custom_config)
                
            generation_config.update({
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            })
            
            # Generate summary
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
            
            # Process and clean output
            decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = self._clean_summary(decoded_text)
            
            return summary
            
        except Exception as e:
            print(f"Error during summary generation: {str(e)}")
            return ""


def main():
    """Example usage of the summarizer."""
    # Initialize summarizer
    summarizer = DeepSeekSummarizer()
    
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
    
    # Generate concise summary
    concise_summary = summarizer.generate_summary(
        input_text, 
        config=ConciseGenerationConfig()
    )
    print("\nConcise Summary:")
    print(concise_summary)
    
    # Generate creative summary
    creative_summary = summarizer.generate_summary(
        input_text, 
        config=CreativeGenerationConfig()
    )
    print("\nCreative Summary:")
    print(creative_summary)


if __name__ == "__main__":
    main()