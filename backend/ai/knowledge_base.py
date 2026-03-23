"""
Knowledge base for mixing rules and instrument profiles.
Uses ChromaDB for vector similarity search, falls back to keyword matching.
"""
import os
import logging
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    """Single knowledge entry."""
    id: str
    content: str
    category: str  # 'mixing_rule', 'instrument', 'osc_reference', 'troubleshooting', 'checklist'
    metadata: Dict[str, str]
    relevance_score: float = 0.0


class KnowledgeBase:
    """Knowledge base with vector search and fallback keyword matching."""

    def __init__(self, knowledge_dir: Optional[str] = None, use_vector_db: bool = True):
        self.knowledge_dir = knowledge_dir or os.path.join(os.path.dirname(__file__), 'knowledge')
        self.entries: List[KnowledgeEntry] = []
        self._collection = None
        self._chroma_client = None
        self._use_vector_db = use_vector_db

        if use_vector_db:
            try:
                import chromadb
                self._chroma_client = chromadb.Client()
                self._collection = self._chroma_client.get_or_create_collection(
                    name="automixer_knowledge",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("ChromaDB vector search initialized")
            except ImportError:
                logger.info("ChromaDB not available, using keyword fallback")
                self._use_vector_db = False
            except Exception as e:
                logger.warning(f"ChromaDB init error: {e}, using keyword fallback")
                self._use_vector_db = False

        self._load_knowledge_files()

    def _load_knowledge_files(self):
        """Load markdown knowledge files from knowledge directory."""
        if not os.path.isdir(self.knowledge_dir):
            logger.warning(f"Knowledge directory not found: {self.knowledge_dir}")
            return

        for filename in sorted(os.listdir(self.knowledge_dir)):
            if not filename.endswith('.md'):
                continue
            filepath = os.path.join(self.knowledge_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                category = filename.replace('.md', '')
                sections = self._split_sections(content, category)
                for section in sections:
                    self.add_entry(section)
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")

        logger.info(f"Loaded {len(self.entries)} knowledge entries from {self.knowledge_dir}")

    def _split_sections(self, content: str, category: str) -> List[KnowledgeEntry]:
        """Split markdown into sections at ## headings."""
        sections = []
        current_title = category
        current_content: List[str] = []

        for line in content.split('\n'):
            if line.startswith('## '):
                if current_content:
                    text = '\n'.join(current_content).strip()
                    if text:
                        entry_id = hashlib.md5(text[:200].encode()).hexdigest()[:12]
                        sections.append(KnowledgeEntry(
                            id=f"{category}_{entry_id}",
                            content=text,
                            category=category,
                            metadata={'title': current_title, 'source': category}
                        ))
                current_title = line[3:].strip()
                current_content = [line]
            else:
                current_content.append(line)

        if current_content:
            text = '\n'.join(current_content).strip()
            if text:
                entry_id = hashlib.md5(text[:200].encode()).hexdigest()[:12]
                sections.append(KnowledgeEntry(
                    id=f"{category}_{entry_id}",
                    content=text,
                    category=category,
                    metadata={'title': current_title, 'source': category}
                ))

        return sections

    def add_entry(self, entry: KnowledgeEntry):
        """Add entry to knowledge base."""
        self.entries.append(entry)
        if self._use_vector_db and self._collection is not None:
            try:
                self._collection.add(
                    ids=[entry.id],
                    documents=[entry.content],
                    metadatas=[entry.metadata]
                )
            except Exception as e:
                logger.debug(f"Vector DB add error: {e}")

    def search(self, query: str, n_results: int = 5, category: Optional[str] = None) -> List[KnowledgeEntry]:
        """Search knowledge base. Returns entries sorted by relevance."""
        if self._use_vector_db and self._collection is not None:
            return self._vector_search(query, n_results, category)
        return self._keyword_search(query, n_results, category)

    def _vector_search(self, query: str, n_results: int, category: Optional[str]) -> List[KnowledgeEntry]:
        """Search using ChromaDB vector similarity."""
        try:
            where_filter = {"source": category} if category else None
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n_results, max(1, len(self.entries))),
                where=where_filter
            )
            matched = []
            if results and results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    for entry in self.entries:
                        if entry.id == doc_id:
                            distance = 0.5
                            if results.get('distances') and results['distances'][0]:
                                distance = results['distances'][0][i]
                            entry.relevance_score = max(0.0, 1.0 - distance)
                            matched.append(entry)
                            break
            return matched
        except Exception as e:
            logger.warning(f"Vector search error: {e}, falling back to keyword")
            return self._keyword_search(query, n_results, category)

    def _keyword_search(self, query: str, n_results: int, category: Optional[str]) -> List[KnowledgeEntry]:
        """Fallback keyword-based search with TF scoring."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored: List[Tuple[float, KnowledgeEntry]] = []

        for entry in self.entries:
            if category and entry.category != category:
                continue
            content_lower = entry.content.lower()
            score = 0.0

            # Exact phrase match bonus
            if query_lower in content_lower:
                score += 10.0

            # Individual word matches weighted by inverse frequency approximation
            for word in query_words:
                if len(word) < 2:
                    continue
                count = content_lower.count(word)
                if count > 0:
                    # Longer words are more specific, weight them higher
                    word_weight = min(3.0, len(word) / 3.0)
                    score += count * word_weight

            # Title match bonus
            title_lower = entry.metadata.get('title', '').lower()
            for word in query_words:
                if word in title_lower:
                    score += 5.0

            if score > 0:
                entry.relevance_score = min(1.0, score / 30.0)
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:n_results]]

    def get_by_category(self, category: str) -> List[KnowledgeEntry]:
        """Get all entries in a category."""
        return [e for e in self.entries if e.category == category]

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get a specific entry by ID."""
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None

    def get_categories(self) -> List[str]:
        """Get all unique categories."""
        return list(set(e.category for e in self.entries))

    def entry_count(self) -> int:
        """Return total number of entries."""
        return len(self.entries)
