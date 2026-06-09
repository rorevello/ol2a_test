"""Reusable search backend for the persisted CORD-19 Qwen/FAISS index."""

import importlib.util
import os
from pathlib import Path
from typing import Dict, List

import ir_datasets
import torch

try:
    from .persistent_index import FaissManager, QwenVectoriser
except ImportError:
    from persistent_index import FaissManager, QwenVectoriser


DEFAULT_QUERY_PROMPT = "Find scientific documents that answer the question"


class PersistentQwenStore:
    def __init__(
        self,
        index_path: Path,
        dataset_name: str = "cord19/fulltext/trec-covid",
        ir_datasets_home: Path = Path(".ir_datasets"),
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "auto",
        query_prompt: str = DEFAULT_QUERY_PROMPT,
    ) -> None:
        self.index_path = Path(index_path)
        self.vectoriser_path = Path(f"{self.index_path}.vect")
        self.dataset_name = dataset_name
        self.query_prompt = query_prompt

        if not self.index_path.is_file():
            raise FileNotFoundError(f"No existe el índice FAISS: {self.index_path}")
        if not self.vectoriser_path.is_file():
            raise FileNotFoundError(
                f"No existe el vectorizador Qwen: {self.vectoriser_path}"
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Se solicitó CUDA, pero PyTorch no detecta una GPU CUDA.")
        self.device = device

        self.manager = FaissManager()
        can_load_pickle = (
            device == "cuda" and importlib.util.find_spec("bitsandbytes") is not None
        )
        if can_load_pickle:
            self.manager.load_index(
                str(self.index_path),
                reload_vectoriser=True,
                vectoriser_device=device,
            )
            self.vectoriser_source = str(self.vectoriser_path)
        else:
            self.manager.load_index(str(self.index_path))
            self.manager.vectoriser = QwenVectoriser(
                model=model_name,
                device=device,
                quantize_4bit=False,
            )
            self.vectoriser_source = f"{model_name} (caché local, {device})"

        self.metadata = self._load_metadata(ir_datasets_home)
        if len(self.metadata) != self.manager.index.ntotal:
            raise RuntimeError(
                "El número de documentos CORD-19 no coincide con el índice: "
                f"{len(self.metadata)} != {self.manager.index.ntotal}."
            )

    def _load_metadata(self, ir_datasets_home: Path) -> Dict[int, Dict[str, str]]:
        os.environ["IR_DATASETS_HOME"] = str(ir_datasets_home)
        dataset = ir_datasets.load(self.dataset_name)
        docs_in_qrels = {qrel.doc_id for qrel in dataset.qrels_iter()}

        metadata: Dict[int, Dict[str, str]] = {}
        seen_doc_ids = set()
        for doc in dataset.docs_iter():
            doc_id = doc[0]
            if doc_id not in docs_in_qrels or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            title = doc[1] or "Sin título"
            abstract = doc[4] or ""
            content_parts = [title, abstract]
            previous_section = ""
            for passage in doc[5]:
                if passage.title != previous_section:
                    content_parts.append(passage.title)
                previous_section = passage.title
                content_parts.append(passage.text)

            position = len(metadata)
            metadata[position] = {
                "doc_id": str(doc_id),
                "title": str(title),
                "text": "\n".join(part for part in content_parts if part),
            }
        return metadata

    @property
    def size(self) -> int:
        return int(self.manager.index.ntotal)

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, object]]:
        if not query or not query.strip():
            return []
        if top_k < 1:
            raise ValueError("top_k debe ser mayor que cero.")

        distances, ids = self.manager.text_search(
            query.strip(),
            prompt=self.query_prompt,
            k=top_k,
        )
        results: List[Dict[str, object]] = []
        for distance, document_index in zip(distances[0], ids[0]):
            if document_index < 0:
                continue
            item = self.metadata[int(document_index)]
            l2_distance = float(distance)
            results.append(
                {
                    "chunk_id": str(document_index),
                    "paper_id": item["doc_id"],
                    "doc_id": item["doc_id"],
                    "title": item["title"],
                    "source_path": (
                        f"ir_datasets:{self.dataset_name}:{item['doc_id']}"
                    ),
                    "section": "full_document",
                    "text": item["text"],
                    "distance_l2": l2_distance,
                    "score": 1.0 / (1.0 + l2_distance),
                }
            )
        return results
