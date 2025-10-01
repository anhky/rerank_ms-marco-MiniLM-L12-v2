import json
import numpy as np
import onnxruntime as ort
from tokenizers import BertWordPieceTokenizer
from scipy.special import expit

MODEL_DIR = "cross-encoder"
ONNX_PATH = f"{MODEL_DIR}/model.onnx"

# --- Load tokenizer ---
with open(f"{MODEL_DIR}/tokenizer_config.json", encoding="utf-8") as f:
    tok_cfg = json.load(f)
with open(f"{MODEL_DIR}/special_tokens_map.json", encoding="utf-8") as f:
    sp_map = json.load(f)

tok = BertWordPieceTokenizer(
    vocab=f"{MODEL_DIR}/vocab.txt",
    lowercase=tok_cfg.get("do_lower_case", True),
    unk_token=sp_map.get("unk_token", "[UNK]")
)
tok.add_special_tokens(list(sp_map.values()))
tok.enable_truncation(max_length=512)
tok.enable_padding(
    pad_id=tok.token_to_id(sp_map["pad_token"]),
    pad_token=sp_map["pad_token"]
)

# --- ONNX session (CPU only) ---
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
input_names = [i.name for i in sess.get_inputs()]

# --- CrossEncoder scoring ---
def ce_score_pairs(pairs, batch_size=32):
    scores = []
    for i in range(0, len(pairs), batch_size):
        encs = tok.encode_batch(pairs[i:i+batch_size])
        feed = {
            "input_ids":      np.array([e.ids for e in encs], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64),
        }
        if any(n in input_names for n in ("token_type_ids", "segment_ids")):
            feed[next(n for n in input_names if n in ("token_type_ids","segment_ids"))] = \
                np.array([e.type_ids for e in encs], dtype=np.int64)

        logits = sess.run(None, feed)[0].squeeze(-1)
        scores.extend(expit(logits).tolist())
    return scores

# --- Demo ---
if __name__ == "__main__":
    query = "how to open a bank account online"
    passages = [
        "Open a checking account with our mobile app in minutes.",
        "We offer car insurance and roadside assistance.",
        "To open an online bank account, prepare your ID and proof of address."
    ]
    pairs = [(query, p) for p in passages]
    for p, s in sorted(zip(passages, ce_score_pairs(pairs)), key=lambda x: x[1], reverse=True):
        print(f"{s:.4f} - {p}")
