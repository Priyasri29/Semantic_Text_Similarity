from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import re

# ---- Text cleaning ----
def clean_text(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---- Cosine similarity helper ----
def compute_cosine(v1, v2):
    return cosine_similarity([v1], [v2])[0][0]

# ---- Request schema ----
class TextPair(BaseModel):
    text1: str
    text2: str

# ---- Initialize FastAPI app ----
app = FastAPI()

# ---- Load models and vectorizer ----
print("🔁 Loading models...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Placeholder for vectorizer and scaler
tfidf_vectorizer = TfidfVectorizer()
scaler = MinMaxScaler()

# Dummy fit to initialize TF-IDF and scaler
# NOTE: Replace this with real data loading if needed
dummy_texts = ["example text one", "example text two"]
tfidf_vectorizer.fit(dummy_texts)
dummy_features = np.array([[0.5, 0.3, 0.2], [0.8, 0.1, 0.1]])
scaler.fit(dummy_features)

# ---- Weights ----
w_sbert, w_use, w_tfidf = 0.7, 0.2, 0.1

@app.post("/predict")
async def predict_similarity(payload: TextPair):
    # Clean texts
    t1 = clean_text(payload.text1)
    t2 = clean_text(payload.text2)

    # SBERT & USE embeddings
    sbert_t1 = sbert_model.encode([t1])[0]
    sbert_t2 = sbert_model.encode([t2])[0]
    use_t1 = use_model([t1])[0].numpy()
    use_t2 = use_model([t2])[0].numpy()

    # TF-IDF
    tfidf_t1 = tfidf_vectorizer.transform([t1])
    tfidf_t2 = tfidf_vectorizer.transform([t2])
    tfidf_sim = compute_cosine(tfidf_t1.toarray()[0], tfidf_t2.toarray()[0])

    # Cosine similarity
    sbert_sim = compute_cosine(sbert_t1, sbert_t2)
    use_sim = compute_cosine(use_t1, use_t2)

    # Normalize (dummy scaler for now)
    sbert_sim_n, use_sim_n, tfidf_sim_n = scaler.transform([[sbert_sim, use_sim, tfidf_sim]])[0]

    # Hybrid score
    hybrid_score = (
        w_sbert * sbert_sim_n +
        w_use * use_sim_n +
        w_tfidf * tfidf_sim_n
    )

    return {"similarity score": round(float(hybrid_score), 4)}
