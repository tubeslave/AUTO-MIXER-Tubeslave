from __future__ import annotations

from datetime import datetime, timezone

from tools.automixer_operator import recovery_governor


def test_allows_single_governed_recovery_for_stale_non_recovery_issue() -> None:
    now = datetime(2026, 5, 17, 9, 0, tzinfo=timezone.utc)
    issues = [
        issue(
            "stale-1",
            "TUB-700",
            "Primary execution issue",
            "in_progress",
            "2026-05-15T00:00:00Z",
        ),
    ]

    report = recovery_governor.evaluate_recovery_governor(
        issues,
        stale_issues=[{"identifier": "TUB-700"}],
        now=now,
    )

    target = report["targets"][0]
    assert target["target_identifier"] == "TUB-700"
    assert target["decision"] == "allow"
    assert report["summary_counts"]["allow"] == 1


def test_suppresses_target_with_existing_open_recovery_issue_during_cooldown() -> None:
    now = datetime(2026, 5, 17, 9, 0, tzinfo=timezone.utc)
    issues = [
        issue(
            "stale-1",
            "TUB-700",
            "Primary execution issue",
            "in_progress",
            "2026-05-15T00:00:00Z",
        ),
        issue(
            "recovery-1",
            "TUB-701",
            "Recover stalled issue TUB-700",
            "todo",
            "2026-05-17T02:00:00Z",
        ),
    ]

    report = recovery_governor.evaluate_recovery_governor(
        issues,
        stale_issues=[{"identifier": "TUB-700"}],
        now=now,
    )

    target = report["targets"][0]
    assert target["decision"] == "suppress"
    assert target["open_recovery_issue_identifiers"] == ["TUB-701"]
    assert target["cooldown_remaining_hours"] == 17.0


def test_manual_review_when_recovery_issue_itself_becomes_stale() -> None:
    now = datetime(2026, 5, 17, 9, 0, tzinfo=timezone.utc)
    issues = [
        issue(
            "recovery-1",
            "TUB-901",
            "Recover stalled issue TUB-700",
            "blocked",
            "2026-05-15T00:00:00Z",
        ),
    ]

    report = recovery_governor.evaluate_recovery_governor(
        issues,
        stale_issues=[{"identifier": "TUB-901"}],
        now=now,
    )

    target = report["targets"][0]
    assert target["decision"] == "manual_review"
    assert target["recursive_chain"] == ["TUB-901", "TUB-700"]
    assert report["summary_counts"]["manual_review"] == 1


def test_manual_review_when_multiple_open_recovery_issues_target_same_issue() -> None:
    now = datetime(2026, 5, 17, 9, 0, tzinfo=timezone.utc)
    issues = [
        issue(
            "stale-1",
            "TUB-700",
            "Primary execution issue",
            "in_progress",
            "2026-05-15T00:00:00Z",
        ),
        issue(
            "recovery-1",
            "TUB-701",
            "Recover stalled issue TUB-700",
            "todo",
            "2026-05-16T01:00:00Z",
        ),
        issue(
            "recovery-2",
            "TUB-702",
            "Recover stalled issue TUB-700",
            "blocked",
            "2026-05-16T04:00:00Z",
        ),
    ]

    report = recovery_governor.evaluate_recovery_governor(
        issues,
        stale_issues=[{"identifier": "TUB-700"}],
        now=now,
    )

    target = report["targets"][0]
    assert target["decision"] == "manual_review"
    assert target["open_recovery_issue_count"] == 2
    assert report["summary_counts"]["duplicate_open_recovery_targets"] == 1


def test_handles_open_recovery_issue_without_activity_timestamp() -> None:
    now = datetime(2026, 5, 17, 9, 0, tzinfo=timezone.utc)
    issues = [
        issue(
            "stale-1",
            "TUB-700",
            "Primary execution issue",
            "in_progress",
            "2026-05-15T00:00:00Z",
        ),
        {
            "id": "recovery-1",
            "identifier": "TUB-701",
            "title": "Recover stalled issue TUB-700",
            "status": "todo",
        },
    ]

    report = recovery_governor.evaluate_recovery_governor(
        issues,
        stale_issues=[{"identifier": "TUB-700"}],
        now=now,
    )

    target = report["targets"][0]
    assert target["decision"] == "suppress"
    assert target["latest_recovery_activity_at"] is None


def issue(issue_id: str, identifier: str, title: str, status: str, last_activity: str) -> dict[str, str]:
    return {
        "id": issue_id,
        "identifier": identifier,
        "title": title,
        "status": status,
        "lastActivityAt": last_activity,
        "updatedAt": last_activity,
        "createdAt": last_activity,
    }
