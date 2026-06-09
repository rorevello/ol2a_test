import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import faiss
import ir_datasets
import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer


SUPPORTED_EXTENSIONS = {".json"}
INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.jsonl"
CONFIG_FILENAME = "config.json"


@dataclass
class ChunkRecord:
    chunk_id: str
    paper_id: str
    title: str
    source_path: str
    text: str
    section: str


@dataclass
class SourceDocument:
    paper_id: str
    title: str
    source_path: str
    sections: List[Dict[str, str]]


def iter_json_files(data_dir: Path, max_files: Optional[int] = None) -> Iterator[Path]:
    count = 0
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        yield path
        count += 1
        if max_files is not None and count >= max_files:
            break


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks


def build_records_from_document(
    document: SourceDocument,
    chunk_size: int,
    chunk_overlap: int,
    content_mode: str = "fulltext",
) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []
    chunk_index = 0
    if content_mode == "abstract_title":
        sections = [
            section
            for section in document.sections
            if section["section"].lower() == "abstract"
        ]
        if not sections and document.title != "Sin titulo":
            sections = [{"section": "title", "text": ""}]
    elif content_mode == "fulltext":
        sections = document.sections
    else:
        raise ValueError(f"content_mode no soportado: {content_mode}")

    for section in sections:
        for chunk in chunk_text(section["text"], chunk_size, chunk_overlap):
            if content_mode == "abstract_title":
                chunk = f"Title: {document.title}\nAbstract: {chunk}"
            records.append(
                ChunkRecord(
                    chunk_id=f"{document.paper_id}-{chunk_index}",
                    paper_id=document.paper_id,
                    title=document.title,
                    source_path=document.source_path,
                    text=chunk,
                    section=section["section"],
                )
            )
            chunk_index += 1
        if content_mode == "abstract_title" and section["section"] == "title":
            records.append(
                ChunkRecord(
                    chunk_id=f"{document.paper_id}-{chunk_index}",
                    paper_id=document.paper_id,
                    title=document.title,
                    source_path=document.source_path,
                    text=f"Title: {document.title}",
                    section="title",
                )
            )
            chunk_index += 1
    return records


def load_json_document(path: Path) -> SourceDocument:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    paper_id = data.get("paper_id") or path.stem
    title = (data.get("metadata") or {}).get("title") or "Sin titulo"
    sections: List[Dict[str, str]] = []

    for abstract in data.get("abstract") or []:
        text = (abstract or {}).get("text", "").strip()
        if text:
            sections.append({"section": "abstract", "text": text})

    for body in data.get("body_text") or []:
        text = (body or {}).get("text", "").strip()
        section_name = (body or {}).get("section") or "body_text"
        if text:
            sections.append({"section": section_name, "text": text})

    return SourceDocument(
        paper_id=paper_id,
        title=title,
        source_path=str(path),
        sections=sections,
    )


def extract_records_from_json(
    path: Path,
    chunk_size: int,
    chunk_overlap: int,
    content_mode: str = "fulltext",
) -> List[ChunkRecord]:
    return build_records_from_document(
        load_json_document(path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        content_mode=content_mode,
    )


def iter_ir_dataset_documents(
    dataset_name: str,
    docs_mode: str,
    ir_datasets_home: Path,
    max_docs: Optional[int] = None,
) -> Iterator[SourceDocument]:
    os.environ["IR_DATASETS_HOME"] = str(ir_datasets_home)
    dataset = ir_datasets.load(dataset_name)

    if docs_mode == "docs_in_qrels":
        qrels = dataset.qrels_dict()
        doc_ids = set()
        for _, docs in qrels.items():
            doc_ids.update(docs.keys())
        doc_iterable = (dataset.docs_store().get(doc_id) for doc_id in sorted(doc_ids))
    elif docs_mode == "all_docs":
        doc_iterable = dataset.docs_iter()
    else:
        raise ValueError(f"docs_mode no soportado: {docs_mode}")

    count = 0
    for doc in doc_iterable:
        source_document = source_document_from_ir_dataset_doc(doc, dataset_name)
        if not source_document.sections:
            continue
        yield source_document
        count += 1
        if max_docs is not None and count >= max_docs:
            break


def source_document_from_ir_dataset_doc(doc: object, dataset_name: str) -> SourceDocument:
    paper_id = getattr(doc, "doc_id", None) or getattr(doc, "cord_uid", None) or "unknown-doc"
    title = getattr(doc, "title", None) or "Sin titulo"

    sections: List[Dict[str, str]] = []
    abstract = getattr(doc, "abstract", None)
    if abstract:
        sections.append({"section": "abstract", "text": str(abstract).strip()})

    body = getattr(doc, "body", None)
    if body:
        sections.append({"section": "body", "text": str(body).strip()})

    if not sections:
        for field in getattr(doc, "_fields", []):
            if field in {"doc_id", "title", "doi", "date", "url", "pubmed_id", "pmcid"}:
                continue
            value = getattr(doc, field, None)
            if value:
                sections.append({"section": field, "text": str(value).strip()})

    return SourceDocument(
        paper_id=str(paper_id),
        title=str(title),
        source_path=f"ir_datasets:{dataset_name}:{paper_id}",
        sections=sections,
    )


class QwenEmbedder:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_seq_length: int = 512,
    ) -> None:
        self.model_name = model_name
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Se solicitó CUDA, pero PyTorch no detecta una GPU CUDA.")

        self.device = device
        model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else None
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=device,
            model_kwargs=model_kwargs,
        )
        self.model.max_seq_length = max_seq_length

    def embed_texts(self, texts: Iterable[str], batch_size: int = 4) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embedding, dtype="float32")


def batched_records(
    records: Iterable[ChunkRecord],
    batch_size: int,
) -> Iterator[List[ChunkRecord]]:
    batch: List[ChunkRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_metadata(output_dir: Path) -> List[Dict[str, str]]:
    metadata_path = output_dir / METADATA_FILENAME
    records = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def write_config(output_dir: Path, config: Dict[str, object]) -> None:
    with (output_dir / CONFIG_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)


def load_config(output_dir: Path) -> Dict[str, object]:
    with (output_dir / CONFIG_FILENAME).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_streaming_index(
    records: Iterable[ChunkRecord],
    output_dir: Path,
    embedding_model_name: str,
    batch_size: int,
    index_batch_size: int,
    device: str,
    max_seq_length: int,
) -> int:
    if batch_size < 1 or index_batch_size < 1:
        raise ValueError("batch_size e index_batch_size deben ser mayores que cero.")

    staging_dir = output_dir.with_name(f"{output_dir.name}.building")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    embedder = QwenEmbedder(
        embedding_model_name,
        device=device,
        max_seq_length=max_seq_length,
    )
    index = None
    chunks_indexed = 0
    metadata_path = staging_dir / METADATA_FILENAME

    try:
        with metadata_path.open("w", encoding="utf-8") as metadata_handle:
            for record_batch in batched_records(records, index_batch_size):
                try:
                    embeddings = embedder.embed_texts(
                        (record.text for record in record_batch),
                        batch_size=batch_size,
                    )
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA se quedó sin memoria generando embeddings. "
                        "Prueba --batch-size 1 o ejecuta con --device cpu."
                    ) from exc

                if index is None:
                    index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)
                for record in record_batch:
                    metadata_handle.write(
                        json.dumps(record.__dict__, ensure_ascii=False) + "\n"
                    )
                chunks_indexed += len(record_batch)
                print(f"\rChunks indexados: {chunks_indexed}", end="", flush=True)

        print()
        if index is None:
            raise ValueError("No se encontraron chunks para indexar.")
        faiss.write_index(index, str(staging_dir / INDEX_FILENAME))
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    output_dir.mkdir(parents=True, exist_ok=True)
    os.replace(staging_dir / INDEX_FILENAME, output_dir / INDEX_FILENAME)
    os.replace(staging_dir / METADATA_FILENAME, output_dir / METADATA_FILENAME)
    staging_dir.rmdir()
    return chunks_indexed


def build_index(
    data_dir: Path,
    output_dir: Path,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    content_mode: str,
    max_files: Optional[int],
    batch_size: int,
    index_batch_size: int,
    device: str,
    max_seq_length: int,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"files_processed": 0}

    def iter_records() -> Iterator[ChunkRecord]:
        for json_path in iter_json_files(data_dir, max_files=max_files):
            records = extract_records_from_json(
                json_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                content_mode=content_mode,
            )
            if records:
                stats["files_processed"] += 1
                yield from records

    chunks_indexed = build_streaming_index(
        iter_records(),
        output_dir=output_dir,
        embedding_model_name=embedding_model_name,
        batch_size=batch_size,
        index_batch_size=index_batch_size,
        device=device,
        max_seq_length=max_seq_length,
    )
    files_processed = stats["files_processed"]
    write_config(
        output_dir,
        {
            "embedding_model": embedding_model_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "content_mode": content_mode,
            "files_processed": files_processed,
            "chunks_indexed": chunks_indexed,
            "max_seq_length": max_seq_length,
        },
    )

    return {
        "files_processed": files_processed,
        "chunks_indexed": chunks_indexed,
        "output_dir": str(output_dir.resolve()),
    }


def build_index_from_ir_dataset(
    dataset_name: str,
    docs_mode: str,
    ir_datasets_home: Path,
    output_dir: Path,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    content_mode: str,
    max_docs: Optional[int],
    batch_size: int,
    index_batch_size: int,
    device: str,
    max_seq_length: int,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ir_datasets_home.mkdir(parents=True, exist_ok=True)

    stats = {"docs_processed": 0}

    def iter_records() -> Iterator[ChunkRecord]:
        for document in iter_ir_dataset_documents(
            dataset_name=dataset_name,
            docs_mode=docs_mode,
            ir_datasets_home=ir_datasets_home,
            max_docs=max_docs,
        ):
            records = build_records_from_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                content_mode=content_mode,
            )
            if records:
                stats["docs_processed"] += 1
                yield from records

    chunks_indexed = build_streaming_index(
        iter_records(),
        output_dir=output_dir,
        embedding_model_name=embedding_model_name,
        batch_size=batch_size,
        index_batch_size=index_batch_size,
        device=device,
        max_seq_length=max_seq_length,
    )
    docs_processed = stats["docs_processed"]
    write_config(
        output_dir,
        {
            "embedding_model": embedding_model_name,
            "dataset_name": dataset_name,
            "docs_mode": docs_mode,
            "ir_datasets_home": str(ir_datasets_home.resolve()),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "content_mode": content_mode,
            "files_processed": docs_processed,
            "chunks_indexed": chunks_indexed,
            "max_seq_length": max_seq_length,
        },
    )

    return {
        "files_processed": docs_processed,
        "chunks_indexed": chunks_indexed,
        "output_dir": str(output_dir.resolve()),
    }


class LocalVectorStore:
    def __init__(self, output_dir: Path, device: str = "auto") -> None:
        self.output_dir = output_dir
        self.config = load_config(output_dir)
        self.index = faiss.read_index(str(output_dir / INDEX_FILENAME))
        self.metadata = load_metadata(output_dir)
        self.embedder = QwenEmbedder(
            str(self.config["embedding_model"]),
            device=device,
            max_seq_length=int(self.config.get("max_seq_length", 512)),
        )
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "El índice FAISS y metadata.jsonl tienen distinto número de registros."
            )

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, object]]:
        query_embedding = self.embedder.embed_query(query)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results


def build_context(
    results: List[Dict[str, object]],
    max_context_chars: int = 24000,
) -> str:
    if not results:
        return "No context was retrieved."

    text_budget = max(1000, (max_context_chars // len(results)) - 300)
    blocks = []
    for i, result in enumerate(results, start=1):
        text = str(result["text"])
        if len(text) > text_budget:
            text = text[:text_budget].rsplit(" ", 1)[0] + "\n[Text truncated]"
        blocks.append(
            "\n".join(
                [
                    f"Result {i}",
                    f"Title: {result['title']}",
                    f"Section: {result['section']}",
                    f"Source: {result['source_path']}",
                    f"Score: {result['score']:.4f}",
                    f"Text: {text}",
                ]
            )
        )
    return "\n\n".join(blocks)[:max_context_chars]


def query_hermes(
    base_url: str,
    model_name: str,
    user_query: str,
    retrieval_results: List[Dict[str, object]],
    timeout: int = 120,
    max_context_chars: int = 24000,
    previous_answer: Optional[str] = None,
    revision_feedback: Optional[str] = None,
) -> str:
    context = build_context(
        retrieval_results,
        max_context_chars=max_context_chars,
    )
    revision_request = ""
    if previous_answer and revision_feedback:
        revision_request = (
            "\n\nThe previous answer was rejected by the validator.\n"
            f"Previous answer:\n{previous_answer}\n\n"
            f"Reason for rejection:\n{revision_feedback}\n\n"
            "Generate a corrected answer that resolves all identified problems."
        )

    payload = {
        "model": model_name,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert assistant for explaining evidence retrieved "
                    "from a scientific vector database. Always answer in English, "
                    "even when the user writes in another language. Use only the "
                    "retrieved evidence, cite document titles when useful, and "
                    "clearly state when the evidence is incomplete or uncertain."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User question:\n{user_query}\n\n"
                    f"Retrieved context:\n{context}\n\n"
                    "Answer the question in English using only the retrieved context."
                    f"{revision_request}"
                ),
            },
        ],
    }

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        detail = response.text.strip()
        if len(detail) > 1000:
            detail = detail[:1000] + "..."
        raise RuntimeError(
            f"Hermes returned HTTP {response.status_code}: "
            f"{detail or 'no error details'}"
        )
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def validate_answer(
    base_url: str,
    model_name: str,
    user_query: str,
    retrieval_results: List[Dict[str, object]],
    candidate_answer: str,
    timeout: int = 120,
    max_context_chars: int = 20000,
) -> Dict[str, object]:
    context = build_context(
        retrieval_results,
        max_context_chars=max_context_chars,
    )
    payload = {
        "model": model_name,
        "temperature": 0.0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict validator of RAG answers. Check that the "
                    "candidate answers the question in English, is supported only "
                    "by the retrieved context, does not invent facts or citations, "
                    "and acknowledges insufficient evidence when appropriate. "
                    "Return JSON only, using this exact structure: "
                    '{"ok": true, "reason": "brief explanation in English"}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{user_query}\n\n"
                    f"Retrieved context:\n{context}\n\n"
                    f"Candidate answer:\n{candidate_answer}"
                ),
            },
        ],
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        detail = response.text.strip()
        if len(detail) > 1000:
            detail = detail[:1000] + "..."
        raise RuntimeError(
            f"The validator returned HTTP {response.status_code}: "
            f"{detail or 'no error details'}"
        )

    content = response.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end < start:
        return {
            "ok": False,
            "reason": f"The validator did not return valid JSON: {content[:300]}",
        }

    try:
        result = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return {
            "ok": False,
            "reason": f"The validator returned malformed JSON: {content[:300]}",
        }

    ok = result.get("ok") is True
    reason = str(result.get("reason") or "The validator did not provide a reason.")
    return {"ok": ok, "reason": reason}


def is_probably_english(text: str) -> bool:
    words = re.findall(r"[a-záéíóúüñ]+", text.lower())
    if not words:
        return False

    spanish_markers = {
        "según",
        "que",
        "del",
        "los",
        "las",
        "una",
        "este",
        "esta",
        "estos",
        "estas",
        "para",
        "por",
        "con",
        "como",
        "mientras",
        "mostró",
        "hallazgos",
        "sugieren",
        "debería",
        "humanos",
        "células",
        "proteína",
        "respuesta",
    }
    english_markers = {
        "the",
        "that",
        "this",
        "these",
        "those",
        "with",
        "from",
        "for",
        "and",
        "was",
        "were",
        "has",
        "have",
        "shows",
        "suggests",
        "evidence",
        "answer",
        "human",
        "cells",
        "protein",
    }
    spanish_count = sum(word in spanish_markers for word in words)
    english_count = sum(word in english_markers for word in words)
    has_spanish_characters = bool(re.search(r"[áéíóúüñ¿¡]", text.lower()))

    if spanish_count >= 4 and spanish_count > english_count:
        return False
    if has_spanish_characters and spanish_count >= 2 and english_count == 0:
        return False
    return True


def query_hermes_validated(
    base_url: str,
    model_name: str,
    user_query: str,
    retrieval_results: List[Dict[str, object]],
    validator_base_url: Optional[str] = None,
    validator_model_name: Optional[str] = None,
    max_attempts: int = 3,
    timeout: int = 120,
) -> Dict[str, object]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be greater than zero.")

    validator_base_url = validator_base_url or base_url
    validator_model_name = validator_model_name or model_name
    previous_answer: Optional[str] = None
    feedback: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        answer = query_hermes(
            base_url=base_url,
            model_name=model_name,
            user_query=user_query,
            retrieval_results=retrieval_results,
            timeout=timeout,
            previous_answer=previous_answer,
            revision_feedback=feedback,
        )
        if not is_probably_english(answer):
            previous_answer = answer
            feedback = (
                "The candidate answer is not written in English. Rewrite the "
                "entire answer in English. Keep only proper names and exact "
                "document titles in their original language when necessary."
            )
            continue

        validation = validate_answer(
            base_url=validator_base_url,
            model_name=validator_model_name,
            user_query=user_query,
            retrieval_results=retrieval_results,
            candidate_answer=answer,
            timeout=timeout,
        )
        if validation["ok"]:
            return {
                "answer": answer,
                "attempts": attempt,
                "validation_reason": validation["reason"],
            }
        previous_answer = answer
        feedback = str(validation["reason"])

    raise RuntimeError(
        f"No answer was approved after {max_attempts} attempts. "
        f"Last validation reason: {feedback}"
    )
