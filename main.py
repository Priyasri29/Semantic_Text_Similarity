from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch

app = FastAPI()

# Load a lightweight SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # or use 'paraphrase-albert-small-v2' for even less memory

# Define input structure
class InputText(BaseModel):
    sentence1: str
    sentence2: str

@app.post("/similarity")
def compute_similarity(data: InputText):
    # SBERT Embedding and cosine similarity
    s1_embed = sbert_model.encode(data.sentence1, convert_to_tensor=True)
    s2_embed = sbert_model.encode(data.sentence2, convert_to_tensor=True)
    sbert_sim = util.pytorch_cos_sim(s1_embed, s2_embed).item()

    return {
        "SBERT_similarity": round(sbert_sim, 4)
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the SBERT similarity API!"}
