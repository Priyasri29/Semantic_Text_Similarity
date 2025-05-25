from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re
import numpy as np

# ------------------------- Setup -------------------------

app = FastAPI(title="Text Similarity API (SBERT + TF-IDF)")

# Load SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Dummy corpus to initialize TF-IDF vectorizer
corpus = [
    "This is a sample sentence.",
    "Another example goes here.",
    "The quick brown fox jumps over the lazy dog."
]
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(corpus)

# MinMaxScaler for normalizing similarity scores
scaler = MinMaxScaler()
scaler.fit([[0, 0], [1, 1]])  # Fit on possible cosine sim range

# ------------------------- Request Model -------------------------

class TextPair(BaseModel):
    text1: str
    text2: str

# ------------------------- Utils -------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_sbert_similarity(t1, t2):
    emb1 = sbert_model.encode([t1])[0]
    emb2 = sbert_model.encode([t2])[0]
    return cosine_similarity([emb1], [emb2])[0][0]

def compute_tfidf_similarity(t1, t2):
    vec1 = tfidf_vectorizer.transform([t1])
    vec2 = tfidf_vectorizer.transform([t2])
    return cosine_similarity(vec1, vec2)[0][0]

# ------------------------- API Route -------------------------

@app.post("/similarity")
def get_similarity(pair: TextPair):
    t1 = clean_text(pair.text1)
    t2 = clean_text(pair.text2)

    sbert_sim = compute_sbert_similarity(t1, t2)
    tfidf_sim = compute_tfidf_similarity(t1, t2)

    # Normalize
    norm_sims = scaler.transform([[sbert_sim, tfidf_sim]])[0]
    sbert_sim_norm = norm_sims[0]
    tfidf_sim_norm = norm_sims[1]

    # Hybrid score
    hybrid_score = 0.85 * sbert_sim_norm + 0.15 * tfidf_sim_norm

    return {
        "sbert_similarity": round(sbert_sim_norm, 4),
        "tfidf_similarity": round(tfidf_sim_norm, 4),
        "hybrid_similarity": round(hybrid_score, 4)
    }

@app.get("/")
def root():
    return {"message": "Welcome to the Text Similarity API (SBERT + TF-IDF)"}
