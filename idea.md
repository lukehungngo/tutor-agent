# AI Tutor Agent Overview

## 📌 Concept
An AI-powered tutor agent designed to help users actively review and evaluate their learning by dynamically generating questions aligned with Bloom’s Taxonomy based on user-provided learning materials (books, course notes, articles, etc.).

---

## 📌 Bloom’s Taxonomy Integration

The tutor agent leverages the cognitive levels defined by Bloom's Taxonomy:

1. **Remember**: Fact-based recall.
2. **Understand**: Explanation-based comprehension.
3. **Apply**: Practical application of concepts.
4. **Analyze**: Connections among concepts.
5. **Evaluate**: Critical assessment or judgment.
6. **Create**: Generation of new ideas or products.

---

## 📌 Architecture (Retrieval-Augmented Generation - RAG)

```text
User Study Material
          │
  ┌───────▼───────┐
  │   Ingestion   │ (Document Loaders, Embeddings, Vector Stores)
  └───────┬───────┘
          │
  ┌───────▼───────┐
  │    Storage    │ (Vector DB like Chroma, Pinecone, FAISS)
  └───────┬───────┘
          │
  ┌───────▼───────┐
  │   Retrieval   │ (Semantic Search, Similarity Search)
  └───────┬───────┘
          │
  ┌───────▼───────┐
  │  Generation   │ (Prompt Templates, LLM Chains)
  └───────┬───────┘
          │
  ┌───────▼───────┐
  │    Output     │ (Bloom’s Taxonomy-based questions)
  └───────────────┘
```

---

## 📌 LangChain Core Functionalities

### 1. **Ingestion & Preprocessing**
- **Document Loaders**: PDF, text, web.
- **Text Chunking**: `RecursiveCharacterTextSplitter`

### 2. **Vector Store & Embeddings**
- **Embeddings**: OpenAI, SentenceTransformers
- **Vector Stores**: Chroma, Pinecone, FAISS

### 3. **Retrieval**
- Semantic and similarity search (`RetrievalQAChain`, `ConversationalRetrievalChain`)

### 4. **Prompt Templates & Generation**
- Bloom-specific prompt templates (customized per cognitive level)
- LLM Chains to dynamically create relevant questions

### Example Prompt Template (Analyze Level)
```python
PromptTemplate(
    input_variables=["context"],
    template="""
    Given the following content:
    {context}

    Generate a question that requires analyzing the relationship between concepts presented above.
    """
)
```

---

## 📌 Recommended Tech Stack

- **Framework:** LangChain (Python)
- **Vector Store:** Chroma/Pinecone
- **LLM/AI Model:** OpenAI GPT-4
- **Backend:** FastAPI
- **Frontend:** React/Next.js or Flutter/React Native
- **Deployment:** Docker, Kubernetes, Cloud platforms

---

## 📌 Additional Features & Scalability

- Adaptive learning paths
- User analytics & performance tracking
- Gamification elements (badges, leaderboards)
- Personalized study recommendations

