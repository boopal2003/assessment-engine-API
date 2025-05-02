from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
from sklearn.preprocessing import normalize

app = FastAPI()

# Load model and data
model = SentenceTransformer("BAAI/bge-small-en")
df = pd.read_csv("shl_assessments_full_data.csv")
embeddings = normalize(np.load("bge_embeddings.npy").astype("float32"), axis=1)

# Setup FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Input schema
class QueryInput(BaseModel):
    query: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend")
def recommend(input: QueryInput):
    query_vec = model.encode([input.query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, 10)
    results = df.iloc[indices[0]]

    recommended = []
    for _, row in results.iterrows():
        recommended.append({
            "url": row.get("Link", ""),
            "adaptive_support": row.get("Adaptive/IRT Support", "No").strip().title() if isinstance(row.get("Adaptive/IRT Support"), str) else "No",
            "description": row.get("Description", "")[:300],
            "duration": int(row.get("Assessment Length", 0)) if str(row.get("Assessment Length", "")).isdigit() else 0,
            "remote_support": row.get("Remote Testing Support", "No").strip().title() if isinstance(row.get("Remote Testing Support"), str) else "No",
            "test_type": [t.strip() for t in row.get("Test Type", "").split(",") if t.strip()]
        })

    return {"recommended_assessments": recommended}
