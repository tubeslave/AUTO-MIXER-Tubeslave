"""Offline A/B/C experiment harness for Decision Engine v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping

from automixer.decision import ActionDecision, ActionPlan, DecisionEngine
from automixer.decision.models import ACTION_NO_ACTION, jsonable
from automixer.knowledge import MixingKnowledgeBase
from automixer.safety import SafetyGate, SafetyGateConfig


PIPELINE_RULES_ONLY = "rules_only"
PIPELINE_RULES_CRITIC = "rules_critic"
PIPELINE_RULES_CRITIC_DECISION = "rules_critic_decision_engine"
PIPELINE_DECISION_DRY_RUN = "decision_engine_dry_run"

PIPELINES = (
    PIPELINE_RULES_ONLY,
    PIPELINE_RULES_CRITIC,
    PIPELINE_RULES_CRITIC_DECISION,
    PIPELINE_DECISION_DRY_RUN,
)


class ExperimentRunner:
    """Run offline v2 comparisons from saved metrics.

    This harness is intentionally metric/action-plan based. It can consume
    outputs from the current offline renderers, rules-only reports, critic
    summaries, or hand-built fixtures without pulling heavy ML training into
    the live path.
    """

    def __init__(
        self,
        *,
        decision_engine: DecisionEngine | None = None,
        safety_config: SafetyGateConfig | Mapping[str, Any] | None = None,
        knowledge_base: MixingKnowledgeBase | None = None,
    ):
        self.knowledge_base = knowledge_base or MixingKnowledgeBase.load()
        self.decision_engine = decision_engine or DecisionEngine(self.knowledge_base)
        self.safety_config = (
            safety_config
            if isinstance(safety_config, SafetyGateConfig)
            else SafetyGateConfig.from_mapping(safety_config)
        )

    def run(
        self,
        input_metrics: Mapping[str, Any],
        output_dir: str | Path,
    ) -> Dict[str, Any]:
        """Run all experiment variants and write JSON + Markdown reports."""
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)

        variants: Dict[str, Dict[str, Any]] = {}
        for pipeline in PIPELINES:
            variants[pipeline] = self._run_pipeline(pipeline, input_metrics)

        report = {
            "generated_at": time.time(),
            "input_metrics": jsonable(input_metrics),
            "variants": variants,
            "diff": _diff_variants(variants),
        }
        json_path = output_path / "experiment_report.json"
        md_path = output_path / "experiment_report.md"
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        md_path.write_text(render_markdown_report(report), encoding="utf-8")
        report["artifacts"] = {
            "json": str(json_path),
            "markdown": str(md_path),
        }
        return report

    def _run_pipeline(self, pipeline: str, input_metrics: Mapping[str, Any]) -> Dict[str, Any]:
        analyzer_output = input_metrics.get("analyzer_output", input_metrics)
        critic_payload = input_metrics.get("critic_evaluations", {})
        current_state = input_metrics.get("current_state", {})
        final_scores = input_metrics.get("final_scores", {})

        if pipeline == PIPELINE_RULES_ONLY:
            plan = _rule_actions_plan(input_metrics)
            if plan is None:
                plan = self.decision_engine.create_action_plan(
                    analyzer_output,
                    {},
                    self.knowledge_base,
                    mode="offline_experiment",
                    plan_id=f"{pipeline}.{int(time.time() * 1000)}",
                )
                plan = _rename_plan(plan, pipeline, ["rules_only_proxy"])
            gate = SafetyGate(self.safety_config)
            gated = gate.evaluate_plan(plan, current_state=current_state, live_mode=False, dry_run=True)
        elif pipeline == PIPELINE_RULES_CRITIC:
            plan = _rule_actions_plan(input_metrics)
            if plan is None:
                plan = self.decision_engine.create_action_plan(
                    analyzer_output,
                    critic_payload,
                    self.knowledge_base,
                    mode="offline_experiment",
                    plan_id=f"{pipeline}.{int(time.time() * 1000)}",
                )
                plan = _rename_plan(plan, pipeline, ["rules_plus_critic_proxy"])
            gate = SafetyGate(self.safety_config)
            gated = gate.evaluate_plan(plan, current_state=current_state, live_mode=False, dry_run=True)
        elif pipeline == PIPELINE_RULES_CRITIC_DECISION:
            plan = self.decision_engine.create_action_plan(
                analyzer_output,
                critic_payload,
                self.knowledge_base,
                mode="offline_experiment",
                plan_id=f"{pipeline}.{int(time.time() * 1000)}",
            )
            gate = SafetyGate(self.safety_config)
            gated = gate.evaluate_plan(plan, current_state=current_state, live_mode=False, dry_run=False)
        else:
            plan = self.decision_engine.create_action_plan(
                analyzer_output,
                critic_payload,
                self.knowledge_base,
                mode="offline_experiment_dry_run",
                plan_id=f"{pipeline}.{int(time.time() * 1000)}",
            )
            dry_config = SafetyGateConfig.from_mapping({**self.safety_config.__dict__, "dry_run": True})
            gate = SafetyGate(dry_config)
            gated = gate.evaluate_plan(plan, current_state=current_state, live_mode=False, dry_run=True)

        return {
            "pipeline": pipeline,
            "input_metrics": jsonable(analyzer_output),
            "action_plan": plan.to_dict(),
            "safety_gate": gated.to_dict(),
            "applied_corrections": [
                decision.to_dict()
                for decision in gated.allowed_plan.decisions
                if decision.action_type != ACTION_NO_ACTION
                and decision.metadata.get("safety_gate", {}).get("send_to_executor")
            ],
            "final_scores": jsonable(final_scores.get(pipeline, {})),
        }


def run_offline_experiment(
    input_metrics_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    payload = json.loads(Path(input_metrics_path).expanduser().read_text(encoding="utf-8"))
    return ExperimentRunner().run(payload, output_dir)


def render_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Offline Decision Engine v2 Experiment",
        "",
        f"Generated at: {report.get('generated_at')}",
        "",
        "## Variants",
        "",
    ]
    for name, variant in report.get("variants", {}).items():
        safety = variant.get("safety_gate", {})
        lines.extend(
            [
                f"### {name}",
                "",
                f"- proposed decisions: {len(variant.get('action_plan', {}).get('decisions', []))}",
                f"- allowed after Safety Gate: {safety.get('allowed_count', 0)}",
                f"- blocked after Safety Gate: {safety.get('blocked_count', 0)}",
                f"- applied corrections: {len(variant.get('applied_corrections', []))}",
                "",
            ]
        )
    lines.extend(["## Diff", ""])
    for pair, diff in report.get("diff", {}).items():
        lines.extend(
            [
                f"### {pair}",
                "",
                f"- added: {', '.join(diff.get('added', [])) or 'none'}",
                f"- removed: {', '.join(diff.get('removed', [])) or 'none'}",
                f"- common: {len(diff.get('common', []))}",
                "",
            ]
        )
    return "\n".join(lines)


def _rule_actions_plan(input_metrics: Mapping[str, Any]) -> ActionPlan | None:
    actions = input_metrics.get("rule_actions")
    if not actions:
        return None
    decisions = [
        ActionDecision.from_dict(item)
        for item in actions
        if isinstance(item, Mapping)
    ]
    return ActionPlan(
        plan_id=f"{PIPELINE_RULES_ONLY}.{int(time.time() * 1000)}",
        mode="offline_experiment",
        decisions=decisions,
        source_modules=["legacy_rules"],
        input_summary={"source": "input_metrics.rule_actions"},
        notes=["Rules-only plan supplied by experiment input."],
    )


def _rename_plan(plan: ActionPlan, plan_id_prefix: str, source_modules: list[str]) -> ActionPlan:
    return ActionPlan(
        plan_id=f"{plan_id_prefix}.{plan.plan_id}",
        created_at=plan.created_at,
        mode=plan.mode,
        decisions=plan.decisions,
        source_modules=[*plan.source_modules, *source_modules],
        input_summary=plan.input_summary,
        notes=plan.notes,
    )


def _diff_variants(variants: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    diff: Dict[str, Dict[str, Any]] = {}
    names = list(variants.keys())
    for left, right in zip(names, names[1:]):
        left_ids = _decision_signatures(variants[left])
        right_ids = _decision_signatures(variants[right])
        diff[f"{left}..{right}"] = {
            "added": sorted(right_ids - left_ids),
            "removed": sorted(left_ids - right_ids),
            "common": sorted(left_ids & right_ids),
        }
    return diff


def _decision_signatures(variant: Mapping[str, Any]) -> set[str]:
    decisions = variant.get("action_plan", {}).get("decisions", [])
    signatures = set()
    for item in decisions:
        action_type = str(item.get("action_type", ""))
        if action_type == ACTION_NO_ACTION:
            continue
        signatures.add(f"{item.get('target')}:{action_type}:{item.get('parameters', {})}")
    return signatures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline Decision Engine v2 experiments")
    parser.add_argument("--input-metrics", required=True, help="JSON file with analyzer/rule/critic metrics")
    parser.add_argument("--output-dir", required=True, help="Directory for experiment_report.json/.md")
    args = parser.parse_args(argv)
    report = run_offline_experiment(args.input_metrics, args.output_dir)
    print(json.dumps({"artifacts": report.get("artifacts", {})}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
