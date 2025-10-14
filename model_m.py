# compare_one_ce_model.py
import os, json, time
from typing import List, Tuple
import numpy as np
import onnxruntime as ort
from scipy.special import expit

# tokenizers
from tokenizers import Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer

###############################################################################
# CẤU HÌNH — sửa cho đúng môi trường của bạn
###############################################################################
# Model A: mmarco-mMiniLMv2-L12-H384-v1 (đa ngôn ngữ)
MODEL_A_DIR  = "./mmarco-mMiniLMv2-L12-H384-v1"     # chứa tokenizer.json / vocab.txt / ...
ONNX_A_PATH  = os.path.join(MODEL_A_DIR, "model_quint8_avx2.onnx")

BATCH_SIZE   = 32
THREADS      = 0           # 0 = để ORT tự chọn
MAX_LENGTH   = 512         # sẽ ưu tiên giá trị trong tokenizer_config.json nếu có


###############################################################################
# TOKENIZER + ONNX
###############################################################################
def _load_tokenizer_from_dir(model_dir: str, max_length: int = 512):
    """
    Thứ tự ưu tiên:
      1) tokenizer.json (fast tokenizer)
      2) BertWordPieceTokenizer + vocab.txt (xem do_lower_case trong tokenizer_config.json)
    Trả về (encode_batch_fn, pad_token_id)
    """
    tok_json = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(tok_json):
        tok = Tokenizer.from_file(tok_json)

        pad_token = "[PAD]"
        sp_map = os.path.join(model_dir, "special_tokens_map.json")
        if os.path.exists(sp_map):
            with open(sp_map, encoding="utf-8") as f:
                m = json.load(f)
                pad_token = m.get("pad_token", pad_token)

        tok.enable_truncation(max_length=max_length)
        tok.enable_padding(direction="right", pad_id=tok.token_to_id(pad_token), pad_token=pad_token)

        def encode_batch(pairs: List[Tuple[str, str]]):
            return tok.encode_batch(pairs)

        return encode_batch, tok.token_to_id(pad_token)

    vocab_path = os.path.join(model_dir, "vocab.txt")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Thiếu tokenizer cho {model_dir} (không có tokenizer.json và vocab.txt).")

    lower_case = True
    unk_token  = "[UNK]"
    pad_token  = "[PAD]"

    tok_cfg = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(tok_cfg):
        with open(tok_cfg, encoding="utf-8") as f:
            cfg = json.load(f)
            lower_case = cfg.get("do_lower_case", lower_case)

    sp_map = os.path.join(model_dir, "special_tokens_map.json")
    if os.path.exists(sp_map):
        with open(sp_map, encoding="utf-8") as f:
            sp = json.load(f)
            unk_token = sp.get("unk_token", unk_token)
            pad_token = sp.get("pad_token", pad_token)

    tok = BertWordPieceTokenizer(vocab=vocab_path, lowercase=lower_case, unk_token=unk_token)

    if os.path.exists(sp_map):
        with open(sp_map, encoding="utf-8") as f:
            sp = json.load(f)
        specials = []
        if isinstance(sp, dict):
            for v in sp.values():
                if isinstance(v, str):
                    specials.append(v)
                elif isinstance(v, dict) and "content" in v:
                    specials.append(v["content"])
        if specials:
            tok.add_special_tokens(specials)

    tok.enable_truncation(max_length=max_length)
    tok.enable_padding(pad_id=tok.token_to_id(pad_token), pad_token=pad_token)

    def encode_batch(pairs: List[Tuple[str, str]]):
        return tok.encode_batch(pairs)

    return encode_batch, tok.token_to_id(pad_token)


def _pick_name(input_names: List[str], *candidates: str):
    for c in candidates:
        if c in input_names:
            return c
    return None


def _make_feed(encs, input_names: List[str]):
    """
    encs: list[Encoding] có .ids, .attention_mask, .type_ids (có thể None)
    """
    ids_name  = _pick_name(input_names, "input_ids", "input_ids:0", "ids", "input.1")
    mask_name = _pick_name(input_names, "attention_mask", "input_mask", "attention_mask:0", "mask", "input.2")
    type_name = _pick_name(input_names, "token_type_ids", "segment_ids", "token_type_ids:0", "type_ids", "input.3")

    feed = {}
    if ids_name:
        feed[ids_name] = np.asarray([e.ids for e in encs], dtype=np.int64)
    if mask_name:
        am = [getattr(e, "attention_mask", None) or [1]*len(e.ids) for e in encs]
        feed[mask_name] = np.asarray(am, dtype=np.int64)
    if type_name:
        ti = [getattr(e, "type_ids", None) or [0]*len(e.ids) for e in encs]
        feed[type_name] = np.asarray(ti, dtype=np.int64)
    return feed


def _build_session(onnx_path: str, threads: int = 0):
    so = ort.SessionOptions()
    if threads and threads > 0:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = max(1, threads // 2)
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    return sess, input_names


def _run_logits(sess: ort.InferenceSession, feed: dict) -> np.ndarray:
    logits = sess.run(None, feed)[0]  # (B,1) hoặc (B,)
    return np.squeeze(logits, axis=-1)


###############################################################################
# LỚP CHO 1 MODEL (tokenizer + onnx)
###############################################################################
class CrossEncoderONNX:
    def __init__(self, model_dir: str, onnx_path: str, threads: int = 0, max_length: int = 512):
        # ưu tiên model_max_length nếu có
        cfg_path = os.path.join(model_dir, "tokenizer_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                    max_length = int(cfg.get("model_max_length", max_length))
            except Exception:
                pass

        self.encode_batch, self.pad_id = _load_tokenizer_from_dir(model_dir, max_length=max_length)
        self.sess, self.input_names    = _build_session(onnx_path, threads=threads)

    def score_pairs(self, pairs: List[Tuple[str, str]], batch_size: int = 32, normalize: str = "sigmoid"):
        scores: List[float] = []
        t_total = 0.0
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i+batch_size]
            encs  = self.encode_batch(chunk)
            feed  = _make_feed(encs, self.input_names)

            t0 = time.perf_counter()
            logits = _run_logits(self.sess, feed)
            t_total += time.perf_counter() - t0

            if normalize == "none":
                scores.extend(logits.tolist())
            else:
                scores.extend(expit(logits).tolist())  # mặc định sigmoid
        return np.array(scores, dtype=np.float32), t_total


###############################################################################
# DEMO
###############################################################################
if __name__ == "__main__":
    query = "điều kiện mở thẻ tín dụng VPBank online"
    passages = [
        "Khách hàng có thể đăng ký mở thẻ tín dụng trực tuyến trên ứng dụng VPBank NEO.",
        "khách hàng phải cần có thu nhập 7 triệu đồng/tháng.",
        "Khách hàng có thể mở hợp đồng tiền gửi online.",
        "Ngân hàng cung cấp chương trình hoàn tiền cho thẻ VPBank Visa Platinum.",
        "Để mở thẻ, khách hàng cần cung cấp bản sao chứng minh thu nhập hoặc sao kê lương.",
        "Thẻ ghi nợ nội địa chỉ sử dụng được trong phạm vi Việt Nam.",
        "VPBank không yêu cầu tài sản thế chấp để phát hành thẻ tín dụng.",
        "Hợp đồng tín dụng có hiệu lực từ ngày ký và được lưu trữ điện tử.",
    ]
    pairs = [(query, p) for p in passages]

    modelA = CrossEncoderONNX(MODEL_A_DIR, ONNX_A_PATH, threads=THREADS, max_length=MAX_LENGTH)
    scores, t_run = modelA.score_pairs(pairs, batch_size=BATCH_SIZE, normalize="sigmoid")

    print("\n=== Điểm từng cặp (Model A) ===")
    print(f"{'Idx':>3} | {'Score':>10} | Passage")
    print("-"*100)
    for i, (s, p) in enumerate(zip(scores, passages)):
        print(f"{i:>3} | {s:>10.4f} | {p}")

    print("\n=== Thời gian suy luận (run only) ===")
    print(f"Model A ({os.path.basename(MODEL_A_DIR)}): {t_run*1000:.2f} ms")
