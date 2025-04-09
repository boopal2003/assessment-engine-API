from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import faiss

app = FastAPI()

# Load data
df = pd.read_csv("shl_assessments_full_data.csv")
embeddings = np.load("bge_embeddings.npy").astype('float32')
embeddings = normalize(embeddings, axis=1)

# FAISS setup
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Model
model = SentenceTransformer("BAAI/bge-small-en")

class QueryInput(BaseModel):
    query: str
    top_k: int = 5

@app.post("/recommend")
def recommend(input: QueryInput):
    query_vec = model.encode([input.query], normalize_embeddings=True).astype('float32')
    scores, indices = index.search(query_vec, input.top_k)
    results = df.iloc[indices[0]]
    
    recommendations = []
    for _, row in results.iterrows():
        recommendations.append({
            "Assessment Name": row["Assessment Name"],
            "Link": row["Link"],
            "Description": row["Description"][:150] + "..." if isinstance(row["Description"], str) else ""
        })
    
    return {"recommendations": recommendations}
