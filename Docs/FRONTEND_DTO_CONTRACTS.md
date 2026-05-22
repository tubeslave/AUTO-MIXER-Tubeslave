# Automixer Frontend DTO Contracts

Date: 2026-05-20

Purpose: display-safe snapshot contracts for the React cockpit. These contracts are read-only UI payloads. They must not contain direct OSC commands, routing write commands, training writes, rule writes, autonomous apply flags, secrets, or console credentials.

The frontend may use runtime-derived fallback data until backend Product Layer payloads are implemented.

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
