# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import re
import uvicorn

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

app = FastAPI(title="Hybrid Text Similarity API")

# Load SBERT model once (caching)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample small corpus for TF-IDF fitting - can be replaced with your own dataset for better vectorizer
corpus = [
    "This is a sample sentence.",
    "Another example sentence for TF-IDF.",
    "Semantic textual similarity using SBERT and TF-IDF.",
    "Lightweight text similarity API."
]

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(corpus)

# Initialize MinMaxScaler - fit with typical min/max [0,1]
scaler = MinMaxScaler()
scaler.fit([[0, 0], [1, 1]])

# Weights for hybrid scoring
W_SBERT = 0.8
W_TFIDF = 0.2

# Pydantic model for input data
class TextPair(BaseModel):
    text1: str
    text2: str

class SimilarityResponse(BaseModel):
    sbert_similarity: float
    tfidf_similarity: float
    hybrid_similarity: float

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)              # Remove URLs
    text = re.sub(r"[^a-z0-9\s]", "", text)          # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()         # Remove extra spaces
    return text

def cosine_sim(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

@app.post("/similarity", response_model=SimilarityResponse)
def compute_similarity(pair: TextPair):
    t1 = clean_text(pair.text1)
    t2 = clean_text(pair.text2)

    if not t1 or not t2:
        raise HTTPException(status_code=400, detail="Both texts must be non-empty after cleaning.")

    # SBERT embeddings
    emb1 = sbert_model.encode(t1)
    emb2 = sbert_model.encode(t2)
    sbert_sim = cosine_sim(emb1, emb2)

    # TF-IDF vectors
    tfidf_vec1 = tfidf_vectorizer.transform([t1]).toarray()[0]
    tfidf_vec2 = tfidf_vectorizer.transform([t2]).toarray()[0]
    tfi
