from .deepseek import DeepSeekSummarizer
from .phi35mini import Phi35MiniSummarizer
from .config import GenerationConfig
from .google_gemma import Gemma3Model

# deepseek = DeepSeekSummarizer()
# phi35mini = Phi35MiniSummarizer()
__all__ = [
    "GenerationConfig",
    "Gemma3Model",
    "DeepSeekSummarizer",
    "Phi35MiniSummarizer",
]
