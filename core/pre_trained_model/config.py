from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class GenerationConfig:
    """Base configuration for text generation."""

    max_new_tokens: int = 150
    min_new_tokens: int = 30
    temperature: float = 0.1  # Very low for consistent outputs
    top_p: float = 0.3  # More focused sampling
    top_k: int = 20  # Limited token selection
    repetition_penalty: float = 1.3
    length_penalty: float = 0.8  # Favor shorter outputs
    num_beams: int = 4  # Increased beam search
    do_sample: bool = True  # Deterministic generation
    early_stopping: bool = True
    no_repeat_ngram_size: int = 3

    def as_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary for model generation."""
        return {k: v for k, v in self.__dict__.items()}
