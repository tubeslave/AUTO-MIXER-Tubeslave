"""Source registry and rule retrieval."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from .logger import SourceDecisionLogger
from .models import (
    DecisionTrace,
    FeedbackRecord,
    RuleMatch,
    SourceGroundedConfig,
    SourceReference,
    SourceRule,
)

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_SOURCES_PATH = DEFAULT_DATA_DIR / "sources.yaml"
DEFAULT_RULES_PATH = DEFAULT_DATA_DIR / "rules.jsonl"


def _tokenize(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if len(token) >= 2}


def _expand_aliases(values: Iterable[str]) -> set[str]:
    aliases = {
        "lead_vocal": {"lead_vocal", "vocal"},
        "backing_vocal": {"backing_vocal", "vocal"},
        "electric_guitar": {"electric_guitar", "guitar"},
        "acoustic_guitar": {"acoustic_guitar", "guitar"},
        "snare_top": {"snare_top", "snare", "drums"},
        "snare_bottom": {"snare_bottom", "snare", "drums"},
        "rack_tom": {"rack_tom", "tom", "drums"},
        "floor_tom": {"floor_tom", "tom", "drums"},
        "overhead": {"overhead", "drums"},
        "kick": {"kick", "drums"},
        "bass_guitar": {"bass_guitar", "bass"},
        "mix_bus": {"mix_bus", "master"},
    }
    expanded: set[str] = set()
    for value in values:
        token = str(value).lower()
        expanded.add(token)
        expanded.update(aliases.get(token, set()))
    return expanded


def _matches_filter(values: Iterable[str], requested: Optional[Iterable[str]]) -> bool:
    requested_set = _expand_aliases(str(item) for item in (requested or []) if str(item).strip())
    if not requested_set:
        return True
    value_set = _expand_aliases(values)
    return bool(value_set & requested_set or "all" in value_set)


class SourceKnowledgeStore:
    """Loads authoritative source metadata and paraphrased rules."""

    def __init__(self, config: Optional[Dict[str, Any] | SourceGroundedConfig] = None):
        self.config = (
            config if isinstance(config, SourceGroundedConfig)
            else SourceGroundedConfig.from_mapping(config)
        )
        self.sources_path = Path(self.config.sources_path) if self.config.sources_path else DEFAULT_SOURCES_PATH
        self.rules_path = Path(self.config.rules_path) if self.config.rules_path else DEFAULT_RULES_PATH
        self.sources: Dict[str, SourceReference] = {}
        self.rules: Dict[str, SourceRule] = {}
        self.reload()

    def reload(self) -> None:
        self.sources = self._load_sources(self.sources_path)
        self.rules = self._load_rules(self.rules_path)

    def _load_sources(self, path: Path) -> Dict[str, SourceReference]:
        if not path.exists():
            return {}
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        rows = payload.get("sources", payload if isinstance(payload, list) else [])
        result: Dict[str, SourceReference] = {}
        for row in rows:
            source = SourceReference.from_mapping(row)
            result[source.source_id] = source
        return result

    def _load_rules(self, path: Path) -> Dict[str, SourceRule]:
        if not path.exists():
            return {}
        result: Dict[str, SourceRule] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rule = SourceRule.from_mapping(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
                result[rule.rule_id] = rule
        return result

    def validate(self) -> List[str]:
        errors: List[str] = []
        for rule in self.rules.values():
            if not rule.source_ids and not self.config.allow_unsourced_rules:
                errors.append(f"{rule.rule_id}: no source_ids")
            for source_id in rule.source_ids:
                if source_id not in self.sources:
                    errors.append(f"{rule.rule_id}: unknown source_id {source_id}")
        return errors

    def get_source(self, source_id: str) -> Optional[SourceReference]:
        return self.sources.get(source_id)

    def get_rule(self, rule_id: str) -> Optional[SourceRule]:
        return self.rules.get(rule_id)

    def search_rules(
        self,
        query: str = "",
        *,
        domains: Optional[Iterable[str]] = None,
        instruments: Optional[Iterable[str]] = None,
        problems: Optional[Iterable[str]] = None,
        action_types: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
        include_inactive: bool = False,
    ) -> List[RuleMatch]:
        limit = int(limit or self.config.default_limit)
        query_tokens = _tokenize(query)
        action_filter = {str(item).lower() for item in (action_types or [])}
        tag_filter = {str(item).lower() for item in (tags or [])}
        matches: List[RuleMatch] = []

        for rule in self.rules.values():
            if not include_inactive and rule.status != "active":
                continue
            if rule.confidence < self.config.min_rule_confidence:
                continue
            if not _matches_filter(rule.domains, domains):
                continue
            if not _matches_filter(rule.instruments, instruments):
                continue
            if not _matches_filter(rule.problems, problems):
                continue
            if tag_filter and not ({item.lower() for item in rule.tags} & tag_filter):
                continue
            if action_filter:
                action_types_in_rule = {
                    str(action.get("action_type", "")).lower()
                    for action in rule.action_templates
                }
                if not (action_types_in_rule & action_filter):
                    continue

            searchable = " ".join(
                [
                    rule.rule_id,
                    rule.summary,
                    rule.rationale,
                    " ".join(rule.domains),
                    " ".join(rule.instruments),
                    " ".join(rule.problems),
                    " ".join(rule.tags),
                    json.dumps(rule.action_templates, sort_keys=True),
                ]
            )
            field_tokens = _tokenize(searchable)
            matched_terms = sorted(query_tokens & field_tokens)
            score = float(len(matched_terms))
            if query and query.lower() in searchable.lower():
                score += 4.0
            if not query_tokens:
                score += 1.0
            score += float(rule.confidence)
            score += 0.4 * len(set(str(item).lower() for item in (domains or [])) & set(rule.domains))
            score += 0.4 * len(set(str(item).lower() for item in (problems or [])) & set(rule.problems))
            matches.append(RuleMatch(rule=rule, relevance_score=score, matched_terms=matched_terms))

        matches.sort(key=lambda item: item.relevance_score, reverse=True)
        return matches[:limit]

    def sources_for_rule(self, rule: SourceRule) -> List[SourceReference]:
        return [self.sources[source_id] for source_id in rule.source_ids if source_id in self.sources]


class SourceKnowledgeLayer:
    """Thin wrapper combining store, retrieval, and optional JSONL logging."""

    def __init__(self, config: Optional[Dict[str, Any] | SourceGroundedConfig] = None):
        self.config = (
            config if isinstance(config, SourceGroundedConfig)
            else SourceGroundedConfig.from_mapping(config)
        )
        self.store = SourceKnowledgeStore(self.config)
        self.logger = SourceDecisionLogger(
            path=self.config.log_path,
            queue_maxsize=self.config.queue_maxsize,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def start(self) -> None:
        if self.enabled:
            self.logger.start()

    def stop(self, timeout: float = 2.0) -> None:
        self.logger.stop(timeout=timeout)

    def retrieve(self, *args: Any, **kwargs: Any) -> List[RuleMatch]:
        matches = self.store.search_rules(*args, **kwargs)
        if self.enabled and self.config.log_retrievals:
            self.logger.log_event(
                "source_rule_retrieval",
                query=args[0] if args else kwargs.get("query", ""),
                matches=[match.to_dict() for match in matches],
            )
        return matches

    def record_decision(self, trace: DecisionTrace) -> bool:
        if not self.enabled:
            return False
        return self.logger.log_decision(trace)

    def record_feedback(self, feedback: FeedbackRecord) -> bool:
        if not self.enabled or not self.config.log_feedback:
            return False
        return self.logger.log_feedback(feedback)
