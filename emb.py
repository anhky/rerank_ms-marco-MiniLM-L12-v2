import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from pydantic import PrivateAttr
from llama_index.core.base.embeddings.base import BaseEmbedding

class EmbeddingGemmaONNXEmbedder(BaseEmbedding):
    """PyPy-safe version for embeddinggemma-300m-ONNX"""

    _model_dir: str = PrivateAttr()
    _onnx_path: str = PrivateAttr()
    _tokenizer_path: str = PrivateAttr()
    _session: ort.InferenceSession = PrivateAttr()
    _tokenizer: Tokenizer = PrivateAttr()
    _prefixes: dict = PrivateAttr()

    def __init__(self, model_dir: str = "embeddinggemma-300m-ONNX", quantized: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._model_dir = model_dir
        self._onnx_path = f"{model_dir}/model_quantized.onnx" if quantized else f"{model_dir}/model.onnx"
        self._tokenizer_path = f"{model_dir}/tokenizer.json"

        self._session = ort.InferenceSession(self._onnx_path, providers=["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(self._tokenizer_path)

        self._prefixes = {
            "query": "task: search result | query: ",
            "document": "title: none | text: ",
        }

    def _normalize(self, v):
        denom = np.sqrt((v * v).sum(axis=1, keepdims=True))
        return v / np.clip(denom, 1e-10, None)

    def _embed(self, texts, prefix_type="document"):
        if isinstance(texts, str):
            texts = [texts]

        prefixed = [self._prefixes[prefix_type] + t for t in texts]
        encoded = self._tokenizer.encode_batch(prefixed)
        max_len = max(len(e.ids) for e in encoded)

        input_ids = np.array([e.ids + [0]*(max_len - len(e.ids)) for e in encoded], dtype=np.int64)
        attention_mask = np.array([[1]*len(e.ids) + [0]*(max_len - len(e.ids)) for e in encoded], dtype=np.int64)

        outputs = self._session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        emb = self._normalize(outputs[1])
        return emb.astype(np.float32)

    def _get_text_embedding(self, text: str) -> np.ndarray:
        return self._embed([text], prefix_type="document")[0]

    def _get_query_embedding(self, query: str) -> np.ndarray:
        return self._embed([query], prefix_type="query")[0]

    async def _aget_query_embedding(self, query: str) -> np.ndarray:
        return self._get_query_embedding(query)
