"""
Knowledge base with vector search for mixing knowledge retrieval.

Supports three backends in priority order:
1. ChromaDB + sentence-transformers (best quality)
2. TF-IDF via sklearn (good quality, no GPU needed)
3. Simple keyword matching (always available)

Thread-safe for concurrent access from multiple agents.
"""

import hashlib
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend availability detection
# ---------------------------------------------------------------------------

_HAS_CHROMA = False
_HAS_SKLEARN = False

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _HAS_CHROMA = True
except ImportError:
    chromadb = None  # type: ignore[assignment]
    logger.info("chromadb not available; will try sklearn TF-IDF fallback")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    _HAS_SKLEARN = True
except ImportError:
    TfidfVectorizer = None  # type: ignore[assignment, misc]
    cosine_similarity = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    logger.info("sklearn not available; will use keyword fallback")


# ---------------------------------------------------------------------------
# Embedding function for ChromaDB (sentence-transformers)
# ---------------------------------------------------------------------------

_HAS_SENTENCE_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment, misc]


class _SentenceTransformerEmbedding:
    """ChromaDB-compatible embedding function using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(input, show_progress_bar=False)
        return embeddings.tolist()


# ---------------------------------------------------------------------------
# ChromaDB backend
# ---------------------------------------------------------------------------

class _ChromaBackend:
    """Vector search via ChromaDB with sentence-transformer embeddings."""

    def __init__(self, persist_dir: Optional[str] = None):
        settings_kwargs: Dict[str, Any] = {
            "anonymized_telemetry": False,
        }
        if persist_dir:
            self._client = chromadb.Client(ChromaSettings(
                persist_directory=persist_dir,
                **settings_kwargs,
            ))
        else:
            self._client = chromadb.Client(ChromaSettings(**settings_kwargs))

        embed_fn = None
        if _HAS_SENTENCE_TRANSFORMERS:
            try:
                embed_fn = _SentenceTransformerEmbedding()
                logger.info("Using sentence-transformers embeddings for ChromaDB")
            except Exception as exc:
                logger.warning(f"Failed to load sentence-transformers model: {exc}")

        col_kwargs: Dict[str, Any] = {"name": "mixing_knowledge"}
        if embed_fn is not None:
            col_kwargs["embedding_function"] = embed_fn

        self._collection = self._client.get_or_create_collection(**col_kwargs)

    def add(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        meta = metadata or {}
        # ChromaDB requires metadata values to be str, int, float, or bool
        safe_meta = {k: v for k, v in meta.items()
                     if isinstance(v, (str, int, float, bool))}
        self._collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[safe_meta],
        )

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count() or 1),
        )
        out: List[Tuple[str, str, float]] = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            docs = results["documents"][0] if results["documents"] else [""] * len(ids)
            dists = results["distances"][0] if results["distances"] else [0.0] * len(ids)
            for doc_id, text, dist in zip(ids, docs, dists):
                # ChromaDB returns L2 distance; convert to similarity score 0-1
                score = 1.0 / (1.0 + dist)
                out.append((doc_id, text, score))
        return out

    @property
    def count(self) -> int:
        return self._collection.count()


# ---------------------------------------------------------------------------
# TF-IDF backend
# ---------------------------------------------------------------------------

class _TfidfBackend:
    """TF-IDF cosine similarity search using sklearn."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
        )
        self._doc_ids: List[str] = []
        self._doc_texts: List[str] = []
        self._matrix = None  # sparse TF-IDF matrix
        self._dirty = True  # needs re-fit

    def add(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        # Update existing or append
        if doc_id in self._doc_ids:
            idx = self._doc_ids.index(doc_id)
            self._doc_texts[idx] = text
        else:
            self._doc_ids.append(doc_id)
            self._doc_texts.append(text)
        self._dirty = True

    def _refit(self) -> None:
        if self._doc_texts:
            self._matrix = self._vectorizer.fit_transform(self._doc_texts)
        self._dirty = False

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        if not self._doc_texts:
            return []
        if self._dirty:
            self._refit()
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[Tuple[str, str, float]] = []
        for idx in top_indices:
            if scores[idx] > 0.0:
                results.append((
                    self._doc_ids[idx],
                    self._doc_texts[idx],
                    float(scores[idx]),
                ))
        return results

    @property
    def count(self) -> int:
        return len(self._doc_ids)


# ---------------------------------------------------------------------------
# Keyword matching backend (always available)
# ---------------------------------------------------------------------------

class _KeywordBackend:
    """Simple keyword matching fallback — no dependencies required."""

    def __init__(self) -> None:
        self._docs: Dict[str, str] = {}  # doc_id -> text

    def add(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._docs[doc_id] = text

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        if not query_words:
            return []

        scored: List[Tuple[str, str, float]] = []
        for doc_id, text in self._docs.items():
            text_lower = text.lower()
            text_words = set(re.findall(r'\w+', text_lower))
            if not text_words:
                continue

            # Jaccard-like score: fraction of query words found in document
            overlap = query_words & text_words
            score = len(overlap) / len(query_words) if query_words else 0.0

            # Bonus for phrase match
            if query_lower in text_lower:
                score += 0.3

            # Bonus for high word overlap ratio
            if text_words:
                coverage = len(overlap) / len(text_words)
                score += coverage * 0.1

            if score > 0.0:
                scored.append((doc_id, text, min(score, 1.0)))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    @property
    def count(self) -> int:
        return len(self._docs)


# ---------------------------------------------------------------------------
# Public KnowledgeBase class
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """
    Thread-safe knowledge base with automatic backend selection.

    Priority:
    1. ChromaDB + sentence-transformers (if available)
    2. sklearn TF-IDF (if sklearn available)
    3. Simple keyword matching (always works)
    """

    def __init__(self, persist_dir: Optional[str] = None):
        """
        Initialize knowledge base.

        Args:
            persist_dir: Directory for ChromaDB persistence (only used if
                         ChromaDB backend is active). None = in-memory.
        """
        self._lock = threading.Lock()
        self._backend_name = "unknown"

        if _HAS_CHROMA:
            try:
                self._backend = _ChromaBackend(persist_dir=persist_dir)
                self._backend_name = "chromadb"
                logger.info("KnowledgeBase using ChromaDB backend")
            except Exception as exc:
                logger.warning(f"ChromaDB init failed ({exc}); falling back to TF-IDF")
                self._backend = self._init_fallback()
        else:
            self._backend = self._init_fallback()

    def _init_fallback(self):
        if _HAS_SKLEARN:
            self._backend_name = "tfidf"
            logger.info("KnowledgeBase using TF-IDF backend")
            return _TfidfBackend()
        else:
            self._backend_name = "keyword"
            logger.info("KnowledgeBase using keyword matching backend")
            return _KeywordBackend()

    @property
    def backend_name(self) -> str:
        """Return name of the active backend."""
        return self._backend_name

    @property
    def document_count(self) -> int:
        """Number of indexed documents."""
        with self._lock:
            return self._backend.count

    def index_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Index a single document.

        Args:
            doc_id: Unique document identifier.
            text: Document text content.
            metadata: Optional metadata dict (e.g. source file, category).
        """
        with self._lock:
            self._backend.add(doc_id, text, metadata)
        logger.debug(f"Indexed document '{doc_id}' ({len(text)} chars)")

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[str, str, float]]:
        """
        Search the knowledge base.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return.

        Returns:
            List of (doc_id, text, score) tuples, sorted by relevance.
        """
        with self._lock:
            results = self._backend.search(query, top_k=top_k)
        return results

    def index_all(self, directory_path: str) -> int:
        """
        Index all .md (Markdown) files in a directory tree.

        Each file is split into sections by headings. Each section becomes
        a separate document with an ID based on the file path and heading.

        Args:
            directory_path: Path to directory containing .md files.

        Returns:
            Number of document chunks indexed.
        """
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            logger.warning(f"Directory not found: {directory_path}")
            return 0

        total = 0
        for md_file in sorted(dir_path.rglob("*.md")):
            try:
                text = md_file.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning(f"Cannot read {md_file}: {exc}")
                continue

            chunks = self._split_markdown(text)
            rel_path = md_file.relative_to(dir_path)

            for i, (heading, body) in enumerate(chunks):
                if not body.strip():
                    continue
                chunk_text = f"{heading}\n{body}" if heading else body
                # Create stable ID from file path and content hash
                content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                doc_id = f"{rel_path}::{i}::{content_hash}"
                metadata = {
                    "source_file": str(rel_path),
                    "heading": heading or "(no heading)",
                    "chunk_index": i,
                }
                self.index_document(doc_id, chunk_text, metadata)
                total += 1

        logger.info(
            f"Indexed {total} chunks from {directory_path} "
            f"(backend={self._backend_name})"
        )
        return total

    @staticmethod
    def _split_markdown(text: str) -> List[Tuple[str, str]]:
        """
        Split markdown text into (heading, body) chunks by ## headings.

        Returns at least one chunk (the preamble before any heading).
        """
        chunks: List[Tuple[str, str]] = []
        current_heading = ""
        current_body_lines: List[str] = []

        for line in text.splitlines():
            if re.match(r'^#{1,3}\s+', line):
                # Save previous chunk
                if current_body_lines or current_heading:
                    chunks.append((current_heading, "\n".join(current_body_lines)))
                current_heading = line.strip()
                current_body_lines = []
            else:
                current_body_lines.append(line)

        # Save final chunk
        if current_body_lines or current_heading:
            chunks.append((current_heading, "\n".join(current_body_lines)))

        if not chunks:
            chunks.append(("", text))

        return chunks
