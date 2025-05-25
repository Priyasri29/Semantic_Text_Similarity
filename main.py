from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load SBERT model (lightweight)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample corpus to fit TF-IDF
sample_corpus = [
    "this is a sample text",
    "another sentence for tfidf training",
    "semantic similarity model",
    "text embeddings for nlp tasks"
]
tfidf = TfidfVectorizer()
tfidf.fit(sample_corpus)

# Define weights
W_SBERT = 0.8
W_TFIDF = 0.2

class InputText(BaseModel):
    text1: str
    text2: str

@app.get("/")
def root():
    return {"message": "Hybrid Semantic Similarity API is up!"}

@app.post("/similarity")
def compute_similarity(data: InputText):
    # Get texts
    t1, t2 = data.text1, data.text2

    # SBERT cosine similarity
    emb1 = sbert_model.encode([t1])[0]
    emb2 = sbert_model.encode([t2])[0]
    sbert_sim = cosine_similarity([emb1], [emb2])[0][0]

    # TF-IDF cosine similarity
    tfidf_vec1 = tfidf.transform([t1])
    tfidf_vec2 = tfidf.transform([t2])
    tfidf_sim = cosine_similarity(tfidf_vec1, tfidf_vec2)[0][0]

    # Hybrid score
    hybrid_score = W_SBERT * sbert_sim + W_TFIDF * tfidf_sim

    return {
        
        "hybrid_similarity": round(hybrid_score, 4)
    }
