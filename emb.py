import time
import logging
import numpy as np
import onnxruntime as ort
import asyncio
import psutil
import os

from typing import List
from tokenizers import Tokenizer
from pydantic import PrivateAttr
from llama_index.core.base.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)

class EmbeddingGemmaONNXEmbedder(BaseEmbedding):
    """High-performance CPU-only Gemma ONNX embedder (optimized)."""

    _model_dir: str = PrivateAttr()
    _onnx_path: str = PrivateAttr()
    _tokenizer_path: str = PrivateAttr()
    _session: ort.InferenceSession = PrivateAttr()
    _tokenizer: Tokenizer = PrivateAttr()
    _output_name: str = PrivateAttr()
    _prefix_enc: dict = PrivateAttr()

    embed_batch_size: int = 64
    max_retries: int = 5
    max_seq_len: int = 512

    def __init__(self, model_dir="embeddinggemma-300m-ONNX", quantized=False, **kwargs):
        super().__init__(**kwargs)
        self._model_dir = model_dir
        self._onnx_path = (
            f"{model_dir}/model_quantized.onnx" if quantized else f"{model_dir}/model.onnx"
        )
        self._tokenizer_path = f"{model_dir}/tokenizer.json"

        # === Session optimization ===
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        cpu_count = psutil.cpu_count(logical=True)
        threads = max(1, cpu_count // 2)
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1

        providers = ["CPUExecutionProvider"]

        t0 = time.time()
        self._session = ort.InferenceSession(self._onnx_path, sess_options=so, providers=providers)
        self._tokenizer = Tokenizer.from_file(self._tokenizer_path)
        self._output_name = self._session.get_outputs()[1].name

        # === Precompute prefix encodings ===
        self._prefix_enc = {
            "query": self._tokenizer.encode("task: search result | query: ").ids,
            "document": self._tokenizer.encode("title: none | text: ").ids,
        }

        # === Warmup ===
        _ = self._session.run([self._output_name],
                              {"input_ids": np.zeros((1, 8), np.int64),
                               "attention_mask": np.ones((1, 8), np.int64)})

        mem = psutil.virtual_memory().percent
        logger.info(
            f"[EmbedGemmaONNX-CPU] Init in {time.time()-t0:.2f}s | threads={threads} | RAM={mem:.1f}%"
        )

    # --------------------- Helpers ---------------------
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-10, None)

    def _encode_batch(self, texts: List[str], prefix_type: str):
        prefix_ids = self._prefix_enc[prefix_type]
        enc = self._tokenizer.encode_batch(texts)
        merged = [prefix_ids + e.ids for e in enc]
        max_len = min(self.max_seq_len, max(len(m) for m in merged))

        n = len(merged)
        input_ids = np.zeros((n, max_len), dtype=np.int64)
        attention_mask = np.zeros((n, max_len), dtype=np.int64)

        for i, m in enumerate(merged):
            l = min(len(m), max_len)
            input_ids[i, :l] = m[:l]
            attention_mask[i, :l] = 1
        return input_ids, attention_mask

    # --------------------- Core Inference ---------------------
    def _embed_batch_once(self, texts: List[str], prefix_type: str) -> np.ndarray:
        input_ids, attention_mask = self._encode_batch(texts, prefix_type)

        for attempt in range(1, self.max_retries + 1):
            try:
                out = self._session.run(
                    [self._output_name],
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                )[0]
                return self._normalize(out).astype(np.float32)
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"[EmbedGemmaONNX] Retry {attempt}/{self.max_retries}: {e}")
                    time.sleep(0.3 * attempt)
                else:
                    raise RuntimeError("ONNX inference failed") from e

    # --------------------- Batching ---------------------
    def _embed_many(self, texts: List[str], prefix_type: str) -> List[List[float]]:
        n = len(texts)
        logger.info(f"[EmbedGemmaONNX] Embedding {n} texts (batch={self.embed_batch_size})")
        t0 = time.time()

        outs = []
        for i in range(0, n, self.embed_batch_size):
            batch = texts[i:i+self.embed_batch_size]
            emb = self._embed_batch_once(batch, prefix_type)
            outs.append(emb)
        arr = np.vstack(outs)

        dt = time.time() - t0
        logger.info(f"→ Done {n} in {dt:.2f}s ({n/dt:.1f} embeds/s)")
        return arr.tolist()

    # --------------------- API ---------------------
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_batch_once([text], "document")[0].tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_batch_once([query], "query")[0].tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed_many(texts, "document")

    def _get_query_embeddings(self, queries: List[str]) -> List[List[float]]:
        return self._embed_many(queries, "query")

    async def _aget_query_embedding(self, query: str) -> List[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_query_embedding, query)

    async def _aget_query_embeddings(self, queries: List[str]) -> List[List[float]]:
        """Async parallel batching for query embeddings."""
        tasks = []
        for i in range(0, len(queries), self.embed_batch_size):
            batch = queries[i:i+self.embed_batch_size]
            tasks.append(asyncio.to_thread(self._embed_batch_once, batch, "query"))
        results = await asyncio.gather(*tasks)
        return np.vstack(results).tolist()
