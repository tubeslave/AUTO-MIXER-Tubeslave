# Source-Grounded Learning

The source-grounded layer is the first step toward learning mix decisions from
authoritative sources while keeping live sound safe. It does not send OSC and it
does not replace the current rule engine. It stores provenance, retrieves
paraphrased atomic rules, and logs every attempted decision for later analysis.

## Why This Exists

Several mix passes produced similar results even when the named method changed.
The next useful step is to learn how to map:

source rule -> audio context -> candidate action -> result -> human feedback

This keeps MERT and Codex useful later as evaluators, not as unattributed sources
of truth.

## Files

- `backend/source_knowledge/data/sources.yaml` - source registry.
- `backend/source_knowledge/data/rules.jsonl` - paraphrased atomic rules.
- `backend/source_knowledge/store.py` - validation and retrieval.
- `backend/source_knowledge/logger.py` - non-blocking JSONL logging.
- `config/source_knowledge.yaml` - standalone example config.
- `logs/source_grounded_decisions.jsonl` - default decision and feedback log.

## Seeded Sources

The initial registry includes Mike Senior, Roey Izhaki, Bobby Owsinski,
Intelligent Music Production, Yamaha Sound Reinforcement Handbook, ITU-R BS.1770,
EBU R128, and several video channels marked as candidate sources for future
reviewed transcript ingestion.

Rules are paraphrases and source metadata only. The repository does not store
long copyrighted excerpts.

## How To Enable

In config:

```yaml
source_knowledge:
  enabled: true
  mode: shadow
  log_path: logs/source_grounded_decisions.jsonl
```

`enabled: true` only enables retrieval/decision logging when code calls this
layer. It does not change OSC behavior by itself.

For the offline multitrack mixer, enable it explicitly:

```bash
PYTHONPATH=backend python tools/offline_agent_mix.py \
  --input-dir "$HOME/Desktop/MIX" \
  --output "$HOME/Desktop/AUTO_MIX_AGENT.mp3" \
  --source-knowledge-enable
```

Use `--source-knowledge-log /path/to/decisions.jsonl` to override the default
JSONL path.

## Example Usage

```python
from source_knowledge import SourceKnowledgeLayer

layer = SourceKnowledgeLayer({"source_knowledge": {"enabled": True}})
layer.start()
matches = layer.retrieve(
    "sharp vocal",
    domains=["eq"],
    instruments=["lead_vocal"],
    problems=["harshness"],
)
layer.stop()
```

## JSONL Events

`source_decision` rows include:

- `session_id`
- `decision_id`
- `channel`
- `instrument`
- `problem`
- `candidate_rule_ids`
- `selected_rule_ids`
- `source_ids`
- `candidate_actions`
- `selected_action`
- `before_metrics`
- `after_metrics`
- `outcome`
- `confidence`
- `osc_sent`
- `safety_state`

`source_feedback` rows attach human feedback to a decision:

- `rating`: `better`, `worse`, `neutral`, or a project-specific label.
- `comment`
- `preferred_action`
- `tags`

## Offline Candidate Rows

`tools/offline_agent_mix.py` now logs EQ bands, compressor settings, pan
placement, FX return buses, and active FX sends when source knowledge is enabled.
The current feedback rows are a Codex listening-proxy note, not a replacement for
operator feedback. Human ratings can be appended later against the same
`session_id` and `decision_id`.

## Next Step

Attach real operator listening feedback after each offline render, then use the
paired source rules, metrics, actions, and feedback as training rows for future
reward/evaluator work. MERT/Codex can later score the same logged rows without
becoming the authority source.
