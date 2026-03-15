"""
Tests for backend.ai.knowledge_base — keyword backend, TF-IDF backend,
KnowledgeBase public API, markdown splitting, and document indexing.

All tests use the keyword backend or in-memory TF-IDF. No external services
(ChromaDB, sentence-transformers) required.
"""

import os
import tempfile

import pytest

from backend.ai.knowledge_base import (
    KnowledgeBase,
    _KeywordBackend,
    _HAS_SKLEARN,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def keyword_backend():
    """Fresh keyword backend."""
    return _KeywordBackend()


@pytest.fixture
def knowledge_base():
    """KnowledgeBase instance (keyword or TF-IDF depending on environment)."""
    return KnowledgeBase()


@pytest.fixture
def populated_kb():
    """KnowledgeBase with a few mixing-related documents indexed."""
    kb = KnowledgeBase()
    kb.index_document("doc1", "Kick drum EQ: boost at 60 Hz for sub weight, cut 400 Hz for boxiness.")
    kb.index_document("doc2", "Snare drum: compress with 4:1 ratio, fast attack, moderate release.")
    kb.index_document("doc3", "Vocal EQ: cut proximity mud around 200 Hz, boost presence at 3 kHz.")
    kb.index_document("doc4", "Bass guitar: use HPF at 30 Hz, slight boost at 80 Hz for warmth.")
    kb.index_document("doc5", "High-pass filter removes rumble and low-frequency noise from channels.")
    return kb


@pytest.fixture
def md_dir():
    """Temporary directory with sample markdown files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        md1 = os.path.join(tmpdir, "mixing_rules.md")
        with open(md1, "w") as f:
            f.write(
                "# Mixing Rules\n\n"
                "## Kick Drum\nBoost sub at 60 Hz. Cut mud at 400 Hz.\n\n"
                "## Snare\nCompress with fast attack. Boost crack at 3.5 kHz.\n\n"
                "## Vocals\nCut proximity effect. Add presence boost.\n"
            )
        md2 = os.path.join(tmpdir, "instrument_profiles.md")
        with open(md2, "w") as f:
            f.write(
                "# Instrument Profiles\n\n"
                "## Bass Guitar\nWarm low end. Cut mud.\n\n"
                "## Acoustic Guitar\nSparkle in the highs. Cut boom.\n"
            )
        yield tmpdir


# ---------------------------------------------------------------------------
# _KeywordBackend tests
# ---------------------------------------------------------------------------

class TestKeywordBackend:

    def test_add_and_count(self, keyword_backend):
        assert keyword_backend.count == 0
        keyword_backend.add("doc1", "Kick drum EQ settings")
        assert keyword_backend.count == 1
        keyword_backend.add("doc2", "Snare compressor settings")
        assert keyword_backend.count == 2

    def test_search_returns_matching_docs(self, keyword_backend):
        keyword_backend.add("doc1", "Kick drum EQ: boost at 60 Hz for sub weight.")
        keyword_backend.add("doc2", "Snare drum: compress with fast attack.")
        results = keyword_backend.search("kick drum", top_k=5)
        assert len(results) > 0
        # First result should be the kick drum doc
        assert results[0][0] == "doc1"

    def test_search_empty_query(self, keyword_backend):
        keyword_backend.add("doc1", "Test content")
        results = keyword_backend.search("", top_k=5)
        assert results == []

    def test_search_no_match(self, keyword_backend):
        keyword_backend.add("doc1", "Kick drum EQ settings")
        results = keyword_backend.search("xyzzy zazzle", top_k=5)
        assert len(results) == 0

    def test_phrase_match_bonus(self, keyword_backend):
        keyword_backend.add("doc1", "high pass filter for kick drum")
        keyword_backend.add("doc2", "filter high frequency noise pass")
        results = keyword_backend.search("high pass filter", top_k=5)
        # doc1 contains the exact phrase, doc2 has the words scattered
        assert len(results) >= 1
        assert results[0][0] == "doc1"
        # The phrase match doc should score higher
        if len(results) >= 2:
            assert results[0][2] >= results[1][2]

    def test_top_k_limits_results(self, keyword_backend):
        for i in range(10):
            keyword_backend.add(f"doc{i}", f"audio mixing EQ compressor document {i}")
        results = keyword_backend.search("audio mixing", top_k=3)
        assert len(results) <= 3

    def test_score_capped_at_one(self, keyword_backend):
        # A very short query that matches the whole doc should still cap at 1.0
        keyword_backend.add("doc1", "eq")
        results = keyword_backend.search("eq", top_k=5)
        for _, _, score in results:
            assert score <= 1.0


# ---------------------------------------------------------------------------
# KnowledgeBase public API tests
# ---------------------------------------------------------------------------

class TestKnowledgeBase:

    def test_backend_name_is_string(self, knowledge_base):
        assert knowledge_base.backend_name in ("chromadb", "tfidf", "keyword")

    def test_initial_document_count_is_zero(self):
        kb = KnowledgeBase()
        assert kb.document_count == 0

    def test_index_and_search(self, populated_kb):
        results = populated_kb.search("kick drum EQ", top_k=3)
        assert len(results) > 0
        # Results are (doc_id, text, score) tuples
        doc_id, text, score = results[0]
        assert isinstance(doc_id, str)
        assert isinstance(text, str)
        assert isinstance(score, float)
        assert score > 0

    def test_document_count_after_indexing(self, populated_kb):
        assert populated_kb.document_count == 5

    def test_search_returns_relevant_results(self, populated_kb):
        results = populated_kb.search("vocal EQ presence", top_k=5)
        assert len(results) > 0
        # At least one result should mention vocals
        texts = [text for _, text, _ in results]
        assert any("vocal" in t.lower() or "presence" in t.lower() for t in texts)

    def test_search_empty_kb(self):
        kb = KnowledgeBase()
        results = kb.search("anything", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Markdown splitting tests
# ---------------------------------------------------------------------------

class TestSplitMarkdown:

    def test_splits_on_headings(self):
        text = "# Title\nIntro\n## Section A\nContent A\n## Section B\nContent B"
        chunks = KnowledgeBase._split_markdown(text)
        assert len(chunks) >= 3  # title + 2 sections

    def test_preserves_heading_text(self):
        text = "## Kick Drum\nBoost at 60 Hz.\n## Snare\nFast attack compression."
        chunks = KnowledgeBase._split_markdown(text)
        headings = [h for h, _ in chunks]
        assert "## Kick Drum" in headings
        assert "## Snare" in headings

    def test_empty_text_returns_one_chunk(self):
        chunks = KnowledgeBase._split_markdown("")
        assert len(chunks) >= 1

    def test_no_headings_returns_full_text(self):
        text = "This is plain text with no markdown headings."
        chunks = KnowledgeBase._split_markdown(text)
        assert len(chunks) == 1
        assert text in chunks[0][1]

    def test_preamble_before_first_heading(self):
        text = "Some preamble text.\n## First Section\nSection content."
        chunks = KnowledgeBase._split_markdown(text)
        # First chunk should be the preamble (no heading)
        assert chunks[0][0] == ""
        assert "preamble" in chunks[0][1]


# ---------------------------------------------------------------------------
# index_all (directory indexing)
# ---------------------------------------------------------------------------

class TestIndexAll:

    def test_indexes_markdown_files(self, md_dir):
        kb = KnowledgeBase()
        count = kb.index_all(md_dir)
        assert count > 0
        assert kb.document_count > 0

    def test_search_after_index_all(self, md_dir):
        kb = KnowledgeBase()
        kb.index_all(md_dir)
        results = kb.search("kick drum", top_k=3)
        assert len(results) > 0

    def test_index_nonexistent_dir(self):
        kb = KnowledgeBase()
        count = kb.index_all("/nonexistent/path/to/nothing")
        assert count == 0

    def test_index_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase()
            count = kb.index_all(tmpdir)
            assert count == 0


# ---------------------------------------------------------------------------
# TF-IDF backend tests (only when sklearn is available)
# ---------------------------------------------------------------------------

class TestTfidfBackend:

    def test_tfidf_add_and_search(self):
        if not _HAS_SKLEARN:
            pytest.skip("sklearn not installed")
        from backend.ai.knowledge_base import _TfidfBackend
        backend = _TfidfBackend()
        backend.add("doc1", "Kick drum equalization for live sound mixing")
        backend.add("doc2", "Snare drum compression techniques for live sound")
        backend.add("doc3", "Vocal microphone technique for clear vocals")
        results = backend.search("kick drum EQ", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "doc1"

    def test_tfidf_empty_search(self):
        if not _HAS_SKLEARN:
            pytest.skip("sklearn not installed")
        from backend.ai.knowledge_base import _TfidfBackend
        backend = _TfidfBackend()
        results = backend.search("anything", top_k=5)
        assert results == []

    def test_tfidf_update_existing_doc(self):
        if not _HAS_SKLEARN:
            pytest.skip("sklearn not installed")
        from backend.ai.knowledge_base import _TfidfBackend
        backend = _TfidfBackend()
        backend.add("doc1", "Old content about kick drums")
        backend.add("doc1", "New content about vocal microphones")
        assert backend.count == 1  # Same doc_id, should replace
        results = backend.search("vocal microphone", top_k=5)
        assert len(results) > 0
        assert results[0][0] == "doc1"
