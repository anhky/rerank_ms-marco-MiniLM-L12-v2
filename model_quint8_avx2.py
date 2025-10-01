import json
import time
import numpy as np
import onnxruntime as ort
from tokenizers import BertWordPieceTokenizer
from scipy.special import expit

MODEL_DIR = "cross-encoder"
ONNX_PATH_1 = f"{MODEL_DIR}/model.onnx"                 # FP/FP16
ONNX_PATH_2 = f"{MODEL_DIR}/model_quint8_avx2.onnx"     # INT8

# ---------------- Tokenizer ----------------
with open(f"{MODEL_DIR}/tokenizer_config.json", encoding="utf-8") as f:
    tok_cfg = json.load(f)
with open(f"{MODEL_DIR}/special_tokens_map.json", encoding="utf-8") as f:
    sp_map = json.load(f)

tok = BertWordPieceTokenizer(
    vocab=f"{MODEL_DIR}/vocab.txt",
    lowercase=tok_cfg.get("do_lower_case", True),
    unk_token=sp_map.get("unk_token", "[UNK]"),
)
tok.add_special_tokens(list(sp_map.values()))
tok.enable_truncation(max_length=512)
tok.enable_padding(
    pad_id=tok.token_to_id(sp_map["pad_token"]),
    pad_token=sp_map["pad_token"],
)

# --------------- Runtime helpers ---------------
def build_session(onnx_path: str, num_threads: int = 0):
    so = ort.SessionOptions()
    if num_threads > 0:
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = max(1, num_threads // 2)
    # CPU-only để tránh cảnh báo provider không tồn tại
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    return sess, input_names

def make_feed(encs, input_names):
    # Chuẩn hóa key cho các biến thể tên input
    def pick(*candidates):
        for c in candidates:
            if c in input_names:
                return c
        return None

    ids_name   = pick("input_ids", "input_ids:0", "ids")
    mask_name  = pick("attention_mask", "input_mask", "attention_mask:0", "mask")
    type_name  = pick("token_type_ids", "segment_ids", "token_type_ids:0")

    feed = {}
    if ids_name:
        feed[ids_name] = np.asarray([e.ids for e in encs], dtype=np.int64)
    if mask_name:
        feed[mask_name] = np.asarray([e.attention_mask for e in encs], dtype=np.int64)
    if type_name:
        # Một số tokenizer có thể không sinh type_ids -> fallback zeros
        type_ids = [getattr(e, "type_ids", None) or [0]*len(e.ids) for e in encs]
        feed[type_name] = np.asarray(type_ids, dtype=np.int64)
    return feed

def run_model(sess, feed):
    logits = sess.run(None, feed)[0]  # (B, 1) hoặc (B,)
    logits = np.squeeze(logits, axis=-1)
    return expit(logits)  # sigmoid -> xác suất

def ce_score_pairs_both(pairs, batch_size=32, threads=0):
    # Init sessions 1 lần
    sess1, in1 = build_session(ONNX_PATH_1, num_threads=threads)
    sess2, in2 = build_session(ONNX_PATH_2, num_threads=threads)

    scores1, scores2 = [], []
    t1_total = 0.0
    t2_total = 0.0

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        # tokenization 1 lần cho cả 2 model
        encs = tok.encode_batch(batch)

        # feed theo từng model (vì tên input có thể khác)
        feed1 = make_feed(encs, in1)
        feed2 = make_feed(encs, in2)

        t0 = time.perf_counter()
        s1 = run_model(sess1, feed1)
        t1_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        s2 = run_model(sess2, feed2)
        t2_total += time.perf_counter() - t0

        scores1.extend(s1.tolist())
        scores2.extend(s2.tolist())

    return np.array(scores1), np.array(scores2), t1_total, t2_total

# --------------- Demo & so sánh ---------------
if __name__ == "__main__":
    query = "how to open a bank account online"
    passages = [
        "Open a checking account with our mobile app in minutes.",
        "We offer car insurance and roadside assistance.",
        "To open an online bank account, prepare your ID and proof of address.",
        "Earn 2% cash back with our credit card.",
    ]
    pairs = [(query, p) for p in passages]

    s1, s2, t1, t2 = ce_score_pairs_both(pairs, batch_size=32, threads=0)

    # In bảng so sánh từng mẫu
    print("\n=== So sánh điểm từng cặp ===")
    print(f"{'Idx':>3} | {'Score FP':>9} | {'Score INT8':>10} | {'AbsΔ':>8} | Passage")
    print("-"*80)
    for idx, (p, a, b) in enumerate(zip(passages, s1, s2)):
        print(f"{idx:>3} | {a:>9.4f} | {b:>10.4f} | {abs(a-b):>8.4f} | {p}")

    # So sánh thời gian
    print("\n=== Thời gian suy luận (chỉ phần run; đã token hóa 1 lần) ===")
    print(f"FP/FP16 model: {t1*1000:.2f} ms")
    print(f"INT8 model  : {t2*1000:.2f} ms")
    if t2 > 0:
        print(f"Tốc độ INT8 / FP = {t1/t2:.2f}x")
