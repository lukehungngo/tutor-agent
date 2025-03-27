import torch
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os

os.environ["MPS_GRAPH_CACHE_DEPTH"] = "2"
os.environ["PYTORCH_MPS_MAX_ALLOC_BUFFER_SIZE"] = "4096"

print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
print("Torch version:", torch.__version__)

warnings.filterwarnings("ignore")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "mps"},  # Remove torch_dtype parameter
    encode_kwargs={
        "batch_size": 128,
        "normalize_embeddings": True,
        "convert_to_numpy": True,  # Ensure NumPy array output
    },
)

# Sample documents
documents = [
    "Machine learning is revolutionizing various industries.",
    "Deep learning models require large amounts of data.",
    "Natural language processing enables better human-computer interaction.",
    "Computer vision systems can analyze visual information.",
] * 32  # Creates 128 documents
# Generate embeddings (returns NumPy array when convert_to_numpy=True)
embeddings_array = embeddings.embed_documents(documents)
# Convert to proper array format for manipulation
if isinstance(embeddings_array, list):
    embeddings_array = np.array(embeddings_array)
# Reshape and calculate similarity
print(embeddings_array.shape)
doc1 = embeddings_array[0].reshape(1, -1)
doc2 = embeddings_array[1].reshape(1, -1)
similarity = cosine_similarity(doc1, doc2)[0][0]
print(f"Embedding dimension: {embeddings_array.shape[1]}")
print(f"Similarity between first two documents: {similarity:.4f}")
