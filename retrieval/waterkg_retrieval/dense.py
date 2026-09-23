"""Dense channel: BGE-large paper embeddings (title + abstract) and exact cosine search."""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL = "BAAI/bge-large-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
VECTORS = "paper_embeddings_bge-large-en-v1.5_fp16.npy"
ROWS = "paper_embeddings_rows.parquet"


class DenseIndex:
    def __init__(self, vector_dir: str | Path, device: str | None = None, chunk: int = 65536):
        import pandas as pd

        vector_dir = Path(vector_dir)
        self.emb = np.load(vector_dir / VECTORS, mmap_mode="r")
        self.ids = pd.read_parquet(vector_dir / ROWS).sort_values("row")["paper_id"].to_numpy()
        if len(self.ids) != self.emb.shape[0]:
            raise ValueError("row table and embedding matrix have different lengths")
        self.row_of = {pid: i for i, pid in enumerate(self.ids)}
        self.chunk = chunk
        self.device = device
        self._model = None

    def encode(self, question: str) -> np.ndarray:
        """CLS pooling + L2 normalisation, with the BGE retrieval instruction prefix."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._model is None:
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._tok = AutoTokenizer.from_pretrained(MODEL)
            self._model = AutoModel.from_pretrained(MODEL).to(dev).eval()
        enc = self._tok([QUERY_PREFIX + question], padding=True, truncation=True, max_length=512,
                        return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            cls = self._model(**enc).last_hidden_state[:, 0]
            q = torch.nn.functional.normalize(cls, p=2, dim=1)
        return q[0].float().cpu().numpy()

    def scores(self, q: np.ndarray) -> np.ndarray:
        out = np.empty(self.emb.shape[0], dtype=np.float32)
        for s in range(0, self.emb.shape[0], self.chunk):
            block = np.asarray(self.emb[s:s + self.chunk], dtype=np.float32)
            out[s:s + block.shape[0]] = block @ q
        return out

    def top(self, all_scores: np.ndarray, n: int) -> list[tuple[str, float]]:
        idx = np.argpartition(-all_scores, n)[:n]
        idx = idx[np.argsort(-all_scores[idx])]
        return [(self.ids[i], float(all_scores[i])) for i in idx]

    def score_of(self, all_scores: np.ndarray, paper_id: str) -> float:
        i = self.row_of.get(paper_id)
        return float(all_scores[i]) if i is not None else float("-inf")
