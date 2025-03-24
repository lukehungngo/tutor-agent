from api import app
import torch
import warnings
import os
import uvicorn
os.environ["MPS_GRAPH_CACHE_DEPTH"] = "2"
os.environ["PYTORCH_MPS_MAX_ALLOC_BUFFER_SIZE"] = "4096"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"    # Don't reserve extra memory
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" 

print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
print("Torch version:", torch.__version__)

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)