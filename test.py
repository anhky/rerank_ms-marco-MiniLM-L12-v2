from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
from scipy.special import expit  # sigmoid

# Đường dẫn tới repo đã clone
REPO_DIR = "ms-marco-MiniLM-L12-v2"
ONNX_PATH = f"{REPO_DIR}/model.onnx"

# 1) Load tokenizer
tok = AutoTokenizer.from_pretrained(REPO_DIR)  # dùng fast tokenizer nếu có

# 2) Tạo phiên ONNXRuntime (CPU). Dùng CUDA nếu có: providers=["CUDAExecutionProvider"]
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

def ce_score_pairs(pairs, max_len=512, apply_sigmoid=True, batch_size=32):
    """
    pairs: list[(query, passage)]
    return: list[float] điểm càng cao càng liên quan
    """
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        texts_a = [q for q,_ in batch]
        texts_b = [p for _,p in batch]
        enc = tok(texts_a, texts_b, padding=True, truncation=True,
                  max_length=max_len, return_tensors="np")

        # ONNX input names có thể là: input_ids, attention_mask, token_type_ids (tùy model)
        inputs = {k: enc[k] for k in sess.get_inputs()[0].name.split()[:0]}  # placeholder
        # Cách an toàn: map theo tên
        feed = {}
        for inp in sess.get_inputs():
            name = inp.name
            if name in enc:
                feed[name] = enc[name]
            elif name == "token_type_ids" and "token_type_ids" not in enc:
                # một số BERT-like cần token_type_ids; nếu tokenizer không trả về thì cấp mảng 0
                feed[name] = np.zeros_like(enc["input_ids"])

        # 3) Run
        outputs = sess.run(None, feed)
        # Thường logits shape: [batch, 1]
        logits = outputs[0].squeeze(-1)
        s = expit(logits) if apply_sigmoid else logits
        scores.extend(s.tolist())
    return scores

# Ví dụ dùng
query = "how to open a bank account online"
passages = [
    "Open a checking account with our mobile app in minutes.",
    "We offer car insurance and roadside assistance.",
    "To open an online bank account, prepare your ID and proof of address."
]
pairs = [(query, p) for p in passages]
scores = ce_score_pairs(pairs)
# Sắp xếp theo điểm giảm dần
ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
for p, s in ranked:
    print(round(s, 4), p)
