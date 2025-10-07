from tokenizers import Tokenizer
import onnxruntime as ort
import numpy as np
from numpy.linalg import norm

# ==== Load model ====
MODEL_A = "embeddinggemma-300m-ONNX"
MODEL_DIR = f"{MODEL_A}/onnx"

MODEL_PATH = f"{MODEL_DIR}/model_quantized.onnx"
TOKENIZER_PATH = f"{MODEL_A}/tokenizer.json"

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# ==== Prefix ====
prefixes = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
}

query = prefixes["query"] + "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]
documents = [prefixes["document"] + x for x in documents]

# ==== Tokenize ====
encoded = tokenizer.encode_batch([query] + documents)
max_len = max(len(e.ids) for e in encoded)

input_ids = np.array([e.ids + [0]*(max_len - len(e.ids)) for e in encoded], dtype=np.int64)
attention_mask = np.array([[1]*len(e.ids) + [0]*(max_len - len(e.ids)) for e in encoded], dtype=np.int64)

# ==== Run inference ====
outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
embeddings = outputs[1]

# ==== Normalize ====
def normalize(v): return v / np.clip(norm(v, axis=1, keepdims=True), 1e-10, None)
embeddings = normalize(embeddings)

query_emb, doc_embs = embeddings[0], embeddings[1:]
scores = np.dot(doc_embs, query_emb)
ranking = np.argsort(scores)[::-1]

print("\n🔎 Query:", query)
print("\n📄 Ranked results:")
for i in ranking:
    print(f"→ {scores[i]:.4f} | {documents[i]}")
