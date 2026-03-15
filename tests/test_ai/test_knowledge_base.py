"""Tests for ai.knowledge_base module."""
import pytest
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from ai.knowledge_base import KnowledgeBase, KnowledgeEntry


class TestKnowledgeBase:
    """Tests for the KnowledgeBase class."""

    def test_init_without_vector_db(self):
        """KnowledgeBase initializes with keyword fallback when vector DB disabled."""
        kb = KnowledgeBase(knowledge_dir="/nonexistent", use_vector_db=False)
        assert kb._use_vector_db is False
        assert kb.entries == []
        assert kb.entry_count() == 0

    def test_add_and_retrieve_entry(self):
        """Entries can be added and retrieved by ID."""
        kb = KnowledgeBase(knowledge_dir="/nonexistent", use_vector_db=False)
        entry = KnowledgeEntry(
            id="test_001",
            content="Lead vocals should sit above the instrument bus for clarity.",
            category="mixing_rule",
            metadata={"title": "Vocal Presence", "source": "mixing_rule"},
        )
        kb.add_entry(entry)
        assert kb.entry_count() == 1
        retrieved = kb.get_entry("test_001")
        assert retrieved is not None
        assert retrieved.content == entry.content
        assert retrieved.category == "mixing_rule"

    def test_keyword_search_finds_matching_entries(self):
        """Keyword search returns entries matching query terms."""
        kb = KnowledgeBase(knowledge_dir="/nonexistent", use_vector_db=False)
        entries = [
            KnowledgeEntry(id="e1", content="Lead vocal microphone placement for live sound",
                           category="instrument", metadata={"title": "Vocals", "source": "instrument"}),
            KnowledgeEntry(id="e2", content="Kick drum microphone placement inside the port",
                           category="instrument", metadata={"title": "Kick Drum", "source": "instrument"}),
            KnowledgeEntry(id="e3", content="High-pass filter on vocals to remove rumble",
                           category="mixing_rule", metadata={"title": "HPF Vocals", "source": "mixing_rule"}),
        ]
        for e in entries:
            kb.add_entry(e)

        results = kb.search("vocal microphone", n_results=5)
        assert len(results) >= 1
        result_ids = [r.id for r in results]
        assert "e1" in result_ids
        # e1 should score highest since it has both "vocal" and "microphone"
        assert results[0].id == "e1"

    def test_get_by_category_filters_correctly(self):
        """get_by_category returns only entries matching the specified category."""
        kb = KnowledgeBase(knowledge_dir="/nonexistent", use_vector_db=False)
        kb.add_entry(KnowledgeEntry(id="a1", content="content a", category="mixing_rule",
                                    metadata={"title": "A", "source": "mixing_rule"}))
        kb.add_entry(KnowledgeEntry(id="b1", content="content b", category="instrument",
                                    metadata={"title": "B", "source": "instrument"}))
        kb.add_entry(KnowledgeEntry(id="a2", content="content c", category="mixing_rule",
                                    metadata={"title": "C", "source": "mixing_rule"}))

        mixing_entries = kb.get_by_category("mixing_rule")
        assert len(mixing_entries) == 2
        instrument_entries = kb.get_by_category("instrument")
        assert len(instrument_entries) == 1

    def test_load_knowledge_from_markdown_files(self):
        """KnowledgeBase loads and splits markdown files from a knowledge directory."""
        tmpdir = tempfile.mkdtemp()
        try:
            md_content = (
                "# Guide\n\n"
                "## Vocal EQ\n"
                "Apply a high-pass filter at 80Hz on lead vocals to reduce rumble.\n\n"
                "## Guitar Placement\n"
                "Pan electric guitars left and right for stereo width.\n"
            )
            with open(os.path.join(tmpdir, "mixing_tips.md"), "w") as f:
                f.write(md_content)

            kb = KnowledgeBase(knowledge_dir=tmpdir, use_vector_db=False)
            assert kb.entry_count() >= 2
            categories = kb.get_categories()
            assert "mixing_tips" in categories
        finally:
            shutil.rmtree(tmpdir)

    def test_keyword_search_with_category_filter(self):
        """Keyword search respects the category filter parameter."""
        kb = KnowledgeBase(knowledge_dir="/nonexistent", use_vector_db=False)
        kb.add_entry(KnowledgeEntry(id="m1", content="Use compression on vocals",
                                    category="mixing_rule",
                                    metadata={"title": "Compression", "source": "mixing_rule"}))
        kb.add_entry(KnowledgeEntry(id="i1", content="Vocal microphone types for compression",
                                    category="instrument",
                                    metadata={"title": "Vocal Mics", "source": "instrument"}))

        results = kb.search("compression vocals", n_results=5, category="mixing_rule")
        assert all(r.category == "mixing_rule" for r in results)

    def test_get_entry_returns_none_for_missing_id(self):
        """get_entry returns None when entry ID does not exist."""
        kb = KnowledgeBase(knowledge_dir="/nonexistent", use_vector_db=False)
        assert kb.get_entry("nonexistent_id") is None
