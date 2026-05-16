from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


RECOVERY_TITLE_RE = re.compile(r"\brecover\s+stalled\s+issue\s+([A-Z]+-\d+)\b", re.IGNORECASE)
OPEN_ISSUE_STATUSES = {"backlog", "todo", "in_progress", "in_review", "blocked"}
ACTIVE_RECOVERY_ACTION_STATUSES = {"active", "open", "running", "queued", "pending", "in_progress"}

DEFAULT_COOLDOWN_HOURS = 24.0
DEFAULT_MAX_OPEN_ATTEMPTS_PER_TARGET = 1


@dataclass(frozen=True)
class RecoveryGovernorThresholds:
    cooldown_hours: float = DEFAULT_COOLDOWN_HOURS
    max_open_attempts_per_target: int = DEFAULT_MAX_OPEN_ATTEMPTS_PER_TARGET

    def as_dict(self) -> dict[str, float | int]:
        return {
            "cooldown_hours": self.cooldown_hours,
            "max_open_attempts_per_target": self.max_open_attempts_per_target,
        }


def evaluate_recovery_governor(
    issues: list[dict[str, Any]],
    *,
    stale_issues: list[dict[str, Any]] | None = None,
    orphaned_issues: list[dict[str, Any]] | None = None,
    now: datetime | None = None,
    thresholds: RecoveryGovernorThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or RecoveryGovernorThresholds()
    now = ensure_aware(now or datetime.now(timezone.utc))

    issue_by_identifier = {
        identifier: issue
        for issue in issues
        if (identifier := issue_identifier(issue))
    }
    recovery_issue_by_identifier = {
        identifier: issue
        for issue in issues
        if (identifier := issue_identifier(issue)) and recovery_target_identifier(issue)
    }

    recovery_groups: dict[str, list[dict[str, Any]]] = {}
    active_fingerprint_targets: dict[str, set[str]] = {}
    for issue in issues:
        target_identifier = recovery_target_identifier(issue)
        if not target_identifier:
            continue
        if normalize_status(issue.get("status") or issue.get("state")) not in OPEN_ISSUE_STATUSES:
            continue
        recovery_groups.setdefault(target_identifier, []).append(issue)

        action = issue.get("activeRecoveryAction")
        if isinstance(action, dict):
            fingerprint = clean_missing(action.get("fingerprint"))
            action_status = normalize_status(action.get("status") or action.get("state"))
            if fingerprint and action_status in ACTIVE_RECOVERY_ACTION_STATUSES:
                active_fingerprint_targets.setdefault(fingerprint, set()).add(target_identifier)

    candidate_identifiers = collect_candidate_identifiers(stale_issues or [], orphaned_issues or [])
    targets = [
        evaluate_target(
            target_identifier,
            issue_by_identifier=issue_by_identifier,
            recovery_issue_by_identifier=recovery_issue_by_identifier,
            recovery_groups=recovery_groups,
            active_fingerprint_targets=active_fingerprint_targets,
            now=now,
            thresholds=thresholds,
        )
        for target_identifier in candidate_identifiers
    ]

    summary_counts = {
        "allow": sum(1 for item in targets if item["decision"] == "allow"),
        "suppress": sum(1 for item in targets if item["decision"] == "suppress"),
        "manual_review": sum(1 for item in targets if item["decision"] == "manual_review"),
        "cooldown_limited": sum(1 for item in targets if item["cooldown_remaining_hours"] > 0),
        "duplicate_open_recovery_targets": sum(
            1 for item in targets if item["open_recovery_issue_count"] > thresholds.max_open_attempts_per_target
        ),
        "recursive_targets": sum(1 for item in targets if len(item["recursive_chain"]) > 1),
    }

    return {
        "thresholds": thresholds.as_dict(),
        "summary_counts": summary_counts,
        "targets": targets,
    }


def collect_candidate_identifiers(
    stale_issues: list[dict[str, Any]],
    orphaned_issues: list[dict[str, Any]],
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in stale_issues + orphaned_issues:
        identifier = clean_missing(item.get("identifier")) or clean_missing(item.get("id"))
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)
        ordered.append(identifier)
    return ordered


def evaluate_target(
    target_identifier: str,
    *,
    issue_by_identifier: dict[str, dict[str, Any]],
    recovery_issue_by_identifier: dict[str, dict[str, Any]],
    recovery_groups: dict[str, list[dict[str, Any]]],
    active_fingerprint_targets: dict[str, set[str]],
    now: datetime,
    thresholds: RecoveryGovernorThresholds,
) -> dict[str, Any]:
    issue = issue_by_identifier.get(target_identifier)
    open_recovery_issues = list(recovery_groups.get(target_identifier, []))
    open_recovery_issues.sort(key=lambda item: issue_activity_time(item) or datetime.min.replace(tzinfo=timezone.utc))

    recovery_chain = build_recovery_chain(target_identifier, recovery_issue_by_identifier)
    recovery_cycle_detected = len(recovery_chain) > 2 and recovery_chain[0] == recovery_chain[-1]
    active_fingerprints = sorted(active_recovery_fingerprints(open_recovery_issues))
    duplicated_fingerprints = [
        fingerprint for fingerprint in active_fingerprints if len(active_fingerprint_targets.get(fingerprint, set())) > 1
    ]
    recovery_activity_times = [
        activity_time
        for recovery_issue in open_recovery_issues
        if (activity_time := issue_activity_time(recovery_issue)) is not None
    ]
    latest_attempt_at = max(recovery_activity_times, default=None)
    cooldown_remaining_hours = cooldown_remaining(latest_attempt_at, now, thresholds.cooldown_hours)

    reasons: list[str] = []
    decision = "allow"

    if len(recovery_chain) > 1:
        decision = "manual_review"
        reason = f"recursive recovery chain detected: {' -> '.join(recovery_chain)}"
        if recovery_cycle_detected:
            reason += " (cycle)"
        reasons.append(reason)

    open_recovery_issue_identifiers = [
        issue_identifier(recovery_issue) or issue_id(recovery_issue) or "unknown"
        for recovery_issue in open_recovery_issues
    ]
    if len(open_recovery_issues) > thresholds.max_open_attempts_per_target:
        decision = "manual_review"
        reasons.append(
            "open recovery attempt ceiling exceeded: "
            + f"{len(open_recovery_issues)} > {thresholds.max_open_attempts_per_target}"
        )
    elif open_recovery_issues:
        decision = "suppress" if decision != "manual_review" else decision
        reasons.append(
            "existing open recovery issue already targets this issue: "
            + ", ".join(open_recovery_issue_identifiers)
        )

    if cooldown_remaining_hours > 0:
        if decision != "manual_review":
            decision = "suppress"
        reasons.append(
            f"cooldown active for {round(cooldown_remaining_hours, 1)}h after the latest recovery attempt"
        )

    if duplicated_fingerprints:
        decision = "manual_review"
        reasons.append(
            "active recovery fingerprint already spans multiple targets: "
            + ", ".join(duplicated_fingerprints)
        )

    if not reasons:
        reasons.append("eligible for a single governed recovery attempt")

    return {
        "target_identifier": target_identifier,
        "target_title": clean_missing(issue.get("title")) if isinstance(issue, dict) else None,
        "target_status": normalize_status(issue.get("status") or issue.get("state")) if isinstance(issue, dict) else None,
        "decision": decision,
        "reasons": reasons,
        "open_recovery_issue_count": len(open_recovery_issues),
        "open_recovery_issue_identifiers": open_recovery_issue_identifiers,
        "active_recovery_fingerprints": active_fingerprints,
        "cooldown_remaining_hours": round(cooldown_remaining_hours, 1),
        "latest_recovery_activity_at": normalize_timestamp(latest_attempt_at),
        "recursive_chain": recovery_chain,
    }


def build_recovery_chain(
    start_identifier: str,
    recovery_issue_by_identifier: dict[str, dict[str, Any]],
) -> list[str]:
    path = [start_identifier]
    seen = {start_identifier}
    current_identifier = start_identifier
    while current_identifier in recovery_issue_by_identifier:
        next_identifier = recovery_target_identifier(recovery_issue_by_identifier[current_identifier])
        if not next_identifier:
            break
        path.append(next_identifier)
        if next_identifier in seen:
            break
        seen.add(next_identifier)
        current_identifier = next_identifier
    return path


def active_recovery_fingerprints(open_recovery_issues: list[dict[str, Any]]) -> set[str]:
    fingerprints: set[str] = set()
    for issue in open_recovery_issues:
        action = issue.get("activeRecoveryAction")
        if not isinstance(action, dict):
            continue
        fingerprint = clean_missing(action.get("fingerprint"))
        action_status = normalize_status(action.get("status") or action.get("state"))
        if fingerprint and action_status in ACTIVE_RECOVERY_ACTION_STATUSES:
            fingerprints.add(fingerprint)
    return fingerprints


def cooldown_remaining(last_attempt_at: datetime | None, now: datetime, cooldown_hours: float) -> float:
    if last_attempt_at is None or cooldown_hours <= 0:
        return 0.0
    elapsed_hours = max((ensure_aware(now) - ensure_aware(last_attempt_at)).total_seconds() / 3600.0, 0.0)
    return max(cooldown_hours - elapsed_hours, 0.0)


def recovery_target_identifier(issue: dict[str, Any]) -> str | None:
    title = clean_missing(issue.get("title"))
    if not title:
        return None
    match = RECOVERY_TITLE_RE.search(title)
    return match.group(1).upper() if match else None


def issue_identifier(issue: dict[str, Any]) -> str | None:
    value = issue.get("identifier") or issue.get("issueIdentifier") or issue.get("issueNumber")
    return clean_missing(value)


def issue_id(issue: dict[str, Any]) -> str | None:
    return clean_missing(issue.get("id"))


def issue_activity_time(issue: dict[str, Any]) -> datetime | None:
    for key in ("lastActivityAt", "updatedAt", "startedAt", "createdAt"):
        value = parse_timestamp(issue.get(key))
        if value is not None:
            return value
    return None


def parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ensure_aware(parsed)


def normalize_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return ensure_aware(value).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_status(value: Any) -> str | None:
    text = clean_missing(value)
    return text.lower() if text else None


def clean_missing(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value
