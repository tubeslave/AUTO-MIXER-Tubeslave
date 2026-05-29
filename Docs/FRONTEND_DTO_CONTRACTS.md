# Automixer Frontend DTO Contracts

Date: 2026-05-29

Purpose: display-safe snapshot contracts for the React cockpit. These contracts are read-only UI payloads. They must not contain direct OSC commands, routing write commands, training writes, rule writes, autonomous apply flags, secrets, or console credentials.

The frontend may use runtime-derived fallback data until backend Product Layer payloads are implemented.

## `decision_traces`

```ts
interface DecisionTracesDto {
  type: "decision_traces";
  generated_at?: string;
  status: string;
  source: string;
  artifact_path?: string;
  summary: {
    trace_count: number;
    applied_count: number;
    blocked_count: number;
    verified_count: number;
  };
  traces: AutomationDecisionTrace[];
}
```

Expected source: append-only `AutomationDecisionTrace` store. This is evidence
only. It must never authorize live apply by itself.

## `safety_monitor`

```ts
interface SafetyMonitorDto {
  type: "safety_monitor";
  generated_at?: string;
  status: string;
  source: string;
  operator_mode: unknown;
  connection: unknown;
  manual_write_supervision: unknown;
  live_write_policy: {
    apply_path?: string;
    policy?: string;
    approved_live_write_kinds: string[];
    requires_approval: boolean;
    direct_writes_allowed: false;
    autonomous_apply_allowed: false;
  };
  proof: {
    last_trace?: AutomationDecisionTrace;
    last_pre_read?: unknown;
    last_readback?: unknown;
    last_rollback?: unknown;
    last_proof_status?: string;
  };
  blocked_surfaces: string[];
  summary: {
    ready: boolean;
    mixer_connected: boolean;
    emergency_stopped: boolean;
    reset_required: boolean;
    trace_count: number;
    blocked_trace_count: number;
  };
}
```

Expected source: runtime safety policy plus trace store. This DTO can display
readiness and blockers only; direct writes and autonomous apply remain false.

## `replay_runs`

```ts
interface ReplayRunsDto {
  type: "replay_runs";
  generated_at?: string;
  status: string;
  source: "offline_replay_artifacts" | string;
  artifact_root: string;
  summary: {
    run_count: number;
    offline_only: true;
    live_write: false;
  };
  runs: {
    run_id: string;
    status: string;
    reason?: string;
    generated_at?: string;
    connection_policy?: string;
    write_policy?: string;
    summary?: Record<string, unknown>;
    trace_count: number;
    offline_only: true;
    live_write: false;
    before_after_available: boolean;
    report_path: string;
    summary_path: string;
    decision_traces_path: string;
    metrics_path: string;
  }[];
}
```

Expected source: offline replay artifacts and metrics. `start_replay_run` may
create artifacts and one offline evidence trace, but it must not send WING/OSC
live writes.

## `channel_intelligence`

```ts
interface ChannelIntelligenceDto {
  type: "channel_intelligence";
  generated_at?: string;
  status: string;
  source: string;
  summary: {
    channel_count: number;
    shadow_only: true;
    live_apply_allowed: false;
  };
  channels: {
    channel: number;
    name: string;
    role: string;
    group: string;
    risk: string;
    gain_ride_state: string;
    recent_trace_count: number;
    live_apply_allowed: false;
  }[];
}
```

Expected source: channel inventory, meter snapshots, and decision traces.
Shadow gain rides stay evidence-only until a separate hardware promotion proof.

## `groups_buses_snapshot`

```ts
interface GroupsBusesSnapshotDto {
  generatedAt?: string;
  source?: string;
  buses: BusRecord[];
}
```

Expected source: channel inventory and bus/group analysis. This is display-only; routing edits remain blocked.

## `memory_knowledge_snapshot`

```ts
interface MemoryKnowledgeSnapshotDto {
  generatedAt?: string;
  source?: string;
  memories: MemoryRecord[];
}
```

Expected source: Product Layer memory/knowledge read model. Frontend must not save memory rows.
Verified/offline decision-memory rows may be included as display-safe memory
records. Shadow-only traces must stay evidence, not durable decision memory.

## `training_sessions_snapshot`

```ts
interface TrainingSessionsSnapshotDto {
  generatedAt?: string;
  source?: string;
  sessions: {
    name: string;
    status: string;
    examples: number;
    score: string;
    updated: string;
  }[];
}
```

Expected source: training/read model. Frontend shows queue state only; no live import or dataset write.

## `rules_library_snapshot`

```ts
interface RulesLibrarySnapshotDto {
  generatedAt?: string;
  source?: string;
  rules: {
    name: string;
    category: string;
    confidence: string;
    state: string;
  }[];
}
```

Expected source: policy/rules read model. Frontend can inspect rules, but cannot write or activate rules.

## `agent_communications_snapshot`

```ts
interface AgentCommunicationsSnapshotDto {
  generatedAt?: string;
  source?: string;
  communications: Communication[];
}
```

Expected source: agent message bus read model. Frontend can inspect/focus messages only.

## Apply Boundary

Supervised apply UX is separate from these DTOs. Any real console apply must use the existing backend supervised proposal gate. The frontend must not introduce direct write endpoints or direct OSC/live write transport.
