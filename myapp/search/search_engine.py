import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from myapp.search.objects import Document
from project_progress.part_2.indexing import create_index
from project_progress.part_3.ranking_algorithms import (
    compute_bm25_index,
    compute_custom_index,
    compute_tfidf_vectors,
    rank_bm25,
    rank_custom_score,
    rank_tfidf_cosine,
)


class SearchEngine:
    """Search facade that wires the Part 2/3 ranking algorithms into the web app."""

    SUPPORTED_ALGOS = ("tfidf", "bm25", "custom")

    def __init__(self, processed_data_path: Optional[str] = None, default_algorithm: str = "tfidf") -> None:
        self.processed_data_path = self._resolve_data_path(processed_data_path)
        self.default_algorithm = default_algorithm.lower()
        if self.default_algorithm not in self.SUPPORTED_ALGOS:
            raise ValueError(f"Unsupported default algorithm '{default_algorithm}'")

        raw_docs = self._load_processed_documents(self.processed_data_path)
        self.processed_docs: List[dict] = self._ensure_tokenized_docs(raw_docs, self.processed_data_path)
        self.inverted_index, _ = create_index(self.processed_docs)
        self.num_docs = len(self.processed_docs)
        self.tfidf_vectors = compute_tfidf_vectors(self.inverted_index, self.processed_docs)
        self.bm25_index, self.pid_to_row, self.pid_order = compute_bm25_index(self.processed_docs)
        self.custom_metadata = compute_custom_index(self.processed_docs)

    def search(
        self,
        search_query: str,
        search_id: Optional[int],
        corpus: Dict[str, Document],
        *,
        algorithm: Optional[str] = None,
        top_k: int = 20,
    ) -> List[dict]:
        """Return ranked documents enriched with metadata from the corpus."""

        query = (search_query or "").strip()
        if not query:
            return []

        algo = (algorithm or self.default_algorithm).lower()
        if algo not in self.SUPPORTED_ALGOS:
            raise ValueError(f"Unsupported algorithm '{algorithm}'")

        ranking = self._run_algorithm(algo, query)
        if not ranking:
            return []

        return self._hydrate_results(ranking[:top_k], corpus, search_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_algorithm(self, algorithm: str, query: str) -> List[Tuple[str, float]]:
        if algorithm == "tfidf":
            return rank_tfidf_cosine(query, self.tfidf_vectors, self.inverted_index, self.num_docs)
        if algorithm == "bm25":
            return rank_bm25(query, self.bm25_index, self.pid_to_row, self.pid_order)
        if algorithm == "custom":
            return rank_custom_score(query, self.tfidf_vectors, self.inverted_index, self.num_docs, self.custom_metadata)
        return []

    def _hydrate_results(
        self,
        ranking: Sequence[Tuple[str, float]],
        corpus: Dict[str, Document],
        search_id: Optional[int],
    ) -> List[dict]:
        enriched: List[dict] = []
        for pid, score in ranking:
            doc = corpus.get(pid)
            if not doc:
                continue
            detail_url = f"doc_details?pid={pid}"
            if search_id is not None:
                detail_url += f"&search_id={search_id}"

            enriched.append(
                {
                    "pid": doc.pid,
                    "title": doc.title,
                    "description": doc.description,
                    "brand": doc.brand,
                    "category": doc.category,
                    "sub_category": doc.sub_category,
                    "out_of_stock": doc.out_of_stock,
                    "selling_price": doc.selling_price,
                    "discount": doc.discount,
                    "actual_price": doc.actual_price,
                    "average_rating": doc.average_rating,
                    "url": detail_url,
                    "product_url": doc.url,
                    "ranking": score,
                }
            )
        return enriched

    @staticmethod
    def _load_processed_documents(path: Path) -> List[dict]:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "items" in data and isinstance(data["items"], list):
                return data["items"]
            return list(data.values())
        raise ValueError("Processed dataset must be a list or dict of documents")

    @staticmethod
    def _ensure_tokenized_docs(docs: List[dict], source_path: Path) -> List[dict]:
        token_fields = ("title_tokens", "description_tokens", "product_details_tokens")
        tokenized = [doc for doc in docs if any(doc.get(field) for field in token_fields)]
        if not tokenized:
            raise ValueError(
                "Processed dataset does not contain tokenized fields. Run project_progress/part_1/preprocess.py "
                f"to generate tokenized data (source: {source_path})."
            )
        return tokenized

    @staticmethod
    def _resolve_data_path(explicit_path: Optional[str]) -> Path:
        project_root = Path(__file__).resolve().parents[2]
        env_candidate = explicit_path or os.getenv("PROCESSED_DATA_PATH") or os.getenv("DATA_FILE_PATH") or "data/processed_fashion.json"

        candidate = Path(env_candidate)
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()

        if not candidate.exists():
            raise FileNotFoundError(f"Processed data file not found: {candidate}")
        return candidate
