# 🎬 Movie Recommendation Assistant

This project demonstrates how to build a **semantic search + AI chat assistant** that helps users find movies based on recommendations.  

It uses:  
- [Qdrant](https://qdrant.tech/) — vector database for storing & searching embeddings  
- [SentenceTransformers](https://www.sbert.net/) — to convert text into embeddings  
- [OpenAI Python SDK](https://github.com/openai/openai-python) — to interact with a **local LLaMA model** running with an OpenAI-compatible API (e.g., [Ollama](https://ollama.ai/) or [llama.cpp server](https://github.com/ggerganov/llama.cpp))  

---

## 🚀 Features
- Store and search text embeddings (movie genres, tags, descriptions, etc.) in Qdrant  
- Query similar movies using semantic search  
- Use a local LLaMA model to **chat with an AI assistant** that recommends movies  

---

## 🛠️ Installation

### 1. Install dependencies
```bash
pip install qdrant-client sentence-transformers openai
```

### 2. Run Qdrant locally
Using Docker:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Run a local LLaMA model
If using **Ollama**:
```bash
ollama run llama3
```

If using **llama.cpp server**:
```bash
./server -m models/llama-3.2-3B-Instruct.Q6_K4_2024-09-24.gguf --port 8080
```

---

## 📂 Example Usage

### 1. Generate embeddings and store in Qdrant
```python
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

client = QdrantClient(host="localhost", port=6333)
encoder = SentenceTransformer("all-MiniLM-L6-v2")

texts = ["Sci-Fi", "Romantic Comedy", "Action Thriller"]
vectors = encoder.encode(texts).tolist()

client.recreate_collection(
    collection_name="movies",
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)

points = [
    models.PointStruct(id=i, vector=vectors[i], payload={"genre": texts[i]})
    for i in range(len(texts))
]

client.upsert(collection_name="movies", points=points)
```

### 2. Query with semantic search
```python
query = "space adventures"
query_vector = encoder.encode([query]).tolist()[0]

results = client.search(collection_name="movies", query_vector=query_vector, limit=3)
for res in results:
    print(res.payload["genre"], res.score)
```

### 3. Chat with local LLaMA model
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-no-key")

search_results = ["Sci-Fi", "Action Thriller"]  # from Qdrant

completion = client.chat.completions.create(
    model="Llama-3.2-3B-Instruct.Q6_K4_2024-09-24",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that helps people find movies."},
        {"role": "user", "content": f"Find some good movies based on these recommendations: {search_results}"}
    ]
)

print(completion.choices[0].message["content"])
```

---

## ⚙️ Configuration
- **Qdrant** default port: `6333`  
- **Local LLaMA server** default port: `8080` (Ollama uses `11434`)  
- `api_key` can be any dummy string if running locally  

---

## 📌 Notes
- Adjust `model` name to match your local LLaMA model file  
- If you’re using Ollama, change `base_url` to:
  ```python
  base_url="http://127.0.0.1:11434/v1"
  ```
- If you want cloud-hosted models, you can swap LLaMA for OpenAI GPT models  

---

## 🧑‍💻 License
MIT License  
