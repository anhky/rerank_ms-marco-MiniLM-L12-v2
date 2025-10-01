import os
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L12-v2"
MINILM_PATH = "ms-marco-MiniLM-L12-v2"
os.makedirs(MINILM_PATH, exist_ok=True)
model = CrossEncoder(MODEL_NAME, cache_folder=MINILM_PATH)
pairs = [(question, n.get_text()) for n in nodes]
scores = model.predict(pairs)