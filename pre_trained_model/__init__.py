from .deepseek import DeepSeekSummarizer
from .phi35mini import Phi35MiniSummarizer
from .config import GenerationConfig

# deepseek = DeepSeekSummarizer()
phi35mini = Phi35MiniSummarizer()
__all__ = ["phi35mini", "GenerationConfig"]
