"""Shared TREC-COVID retrieval evaluation for CLI and Gradio."""

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import requests


DEFAULT_TOPICS_URL = "https://ir.nist.gov/covidSubmit/data/topics-rnd5.xml"


@dataclass
class Topic:
    query_id: str
    title: str
    description: str
    narrative: str


def query_text(query: object) -> str:
    for field in ("narrative", "description", "title"):
        value = getattr(query, field, None)
        if value and str(value).strip():
            return str(value).strip()
    raise ValueError(f"La query {getattr(query, 'query_id', '?')} no contiene texto.")


def load_topics(
    topics_file: Path,
    topics_url: str = DEFAULT_TOPICS_URL,
) -> List[Topic]:
    if not topics_file.is_file():
        topics_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(topics_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"No se pudieron descargar los topics desde {topics_url}. "
                f"También puedes proporcionar el archivo en {topics_file}. Error: {exc}"
            ) from exc
        topics_file.write_bytes(response.content)

    try:
        root = ET.parse(topics_file).getroot()
    except (ET.ParseError, OSError) as exc:
        raise RuntimeError(f"No se pudo leer {topics_file}: {exc}") from exc

    topics = []
    for element in root.findall(".//topic"):
        query_id = element.get("number") or element.get("id")
        if not query_id:
            continue
        topics.append(
            Topic(
                query_id=str(query_id),
                title=(element.findtext("query") or "").strip(),
                description=(element.findtext("question") or "").strip(),
                narrative=(element.findtext("narrative") or "").strip(),
            )
        )
    if not topics:
        raise RuntimeError(f"No se encontraron topics en {topics_file}.")
    return topics


def load_dataset(dataset_name: str, ir_datasets_home: Path):
    os.environ["IR_DATASETS_HOME"] = str(ir_datasets_home)
    import ir_datasets

    return ir_datasets.load(dataset_name)


def evaluate_store(
    store: object,
    dataset: object,
    topics: Iterable[Topic],
    top_k: int = 100,
    run_id: str = "qwen-0.6b",
    calculate_metrics: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, object]:
    topic_list = list(topics)
    rows = []
    scored_documents = []
    for position, topic in enumerate(topic_list, start=1):
        results = store.search(query_text(topic), top_k=top_k)
        for rank, result in enumerate(results, start=1):
            score = float(result["score"])
            doc_id = str(result["doc_id"])
            scored_documents.append((topic.query_id, doc_id, score))
            rows.append(
                [
                    topic.query_id,
                    "Q0",
                    doc_id,
                    rank,
                    f"{score:.8f}",
                    run_id,
                ]
            )
        if progress_callback is not None:
            progress_callback(position, len(topic_list))

    metrics = {}
    if calculate_metrics:
        try:
            import ir_measures
            from ir_measures import NDCG, P, RR, ScoredDoc
        except ImportError as exc:
            raise RuntimeError(
                "Falta ir-measures. Instálalo con: "
                "python -m pip install -r requirements.txt"
            ) from exc
        run = [
            ScoredDoc(query_id, doc_id, score)
            for query_id, doc_id, score in scored_documents
        ]
        calculated = ir_measures.calc_aggregate(
            [NDCG @ 10, P @ 10, RR],
            dataset.qrels_iter(),
            run,
        )
        metrics = {
            str(measure): float(value)
            for measure, value in calculated.items()
        }

    return {
        "query_count": len(topic_list),
        "document_count": len(scored_documents),
        "metrics": metrics,
        "rows": rows,
    }
