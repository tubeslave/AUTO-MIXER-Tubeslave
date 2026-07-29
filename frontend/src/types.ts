import type { ReactNode } from 'react';

export type Tone = 'green' | 'blue' | 'purple' | 'yellow' | 'red' | 'muted';

export type AutomixerMode = 'Analyze' | 'Assist' | 'Supervised' | 'Auto';

export type BackendConnectionState = 'mock' | 'connecting' | 'connected' | 'offline' | 'error';

export type BackendDataState = 'mock_fallback' | 'connecting' | 'connected' | 'partial' | 'disconnected' | 'error' | 'stale';

export interface BackendUiState {
  connectionState: BackendConnectionState;
  dataState: BackendDataState;
  dataStateLabel: string;
  dataStateTone: Tone;
  hasRuntimeData: boolean;
  isPartialData: boolean;
  staleSnapshot: boolean;
  sourceLabel: string;
  connectionLabel: string;
  statusLabel: string;
  statusTone: Tone;
  canCreateProposals: boolean;
  canApplyToConsole: boolean;
  requiresApproval: boolean;
  liveWritePolicy: string;
  applyPath: string;
  writeScope: string[];
  mixerConnected: boolean;
  mixerTransportLocked: boolean;
  observationOnly: boolean;
  audioCaptureRunning: boolean;
  audioCaptureChannels: number;
  audioSampleRate: number;
  availableModes: AutomixerMode[];
  lastUpdated: string;
  message: string;
  proposalApplyResult?: ProposalApplyResult | null;
  transportTerminalClose?: TransportTerminalClose | null;
}

export interface TransportTerminalClose {
  code: number;
  reason: string;
  manualRetryRequired: boolean;
}

export interface SoundcheckRuntime {
  running: boolean;
  state: string;
  message: string;
  error: string;
  observeOnly: boolean;
  proposalOnly: boolean;
  mixerConnected: boolean;
  audioRunning: boolean;
  selectedChannels: number[];
  channelsWithSignal: number;
  totalChannels: number;
  recommendationCount: number;
  transactionPlanCount: number;
  transactionOperationCount: number;
  liveAllowedFamilies: string[];
  familyLifecycle: Record<string, string>;
  familyLiveWriteKinds: Record<string, string[]>;
  familyWriteKinds: Record<string, string[]>;
  familyContracts: Record<string, SoundcheckFamilyContract>;
  derivationComplete: boolean;
  readiness: SoundcheckReadiness;
  replan: SoundcheckReplan;
  interceptedWriteCount: number;
  lastUpdated: string;
}

export interface SoundcheckReplan {
  status: string;
  draftAvailable: boolean;
  draftId: string;
  runId: string;
  fromPlanRevision: number;
  toPlanRevision: number;
  splitOperationCount: number;
  safeAlternativeOperationCount: number;
  deferredTargetCount: number;
  operatorRequests: Array<{
    operationId: string;
    reason: string;
    safeAutomaticReplacementAvailable: boolean;
    selectedSafeStrategy: string;
  }>;
  safeAlternatives: Array<{
    strategy: string;
    translationDb?: number;
    finalTargetVectorPreserved?: boolean;
    relativeBalancePreserved?: boolean;
    currentStepTransitional?: boolean;
    constructiveLaneUsed?: boolean;
  }>;
}

export interface SoundcheckReadinessStage {
  required: number;
  resolved: number;
  applied: number;
  alreadySatisfied: number;
  replaced: number;
  operatorReview: number;
  forbidden: number;
  pending: number;
  ready: boolean;
}

export interface SoundcheckReadiness {
  schemaVersion: string;
  runId: string;
  planRevision: number;
  planFingerprint: string;
  mixReadiness: 'soundcheck_incomplete' | 'safe_core_ready' | 'show_mix_ready';
  safetyState: string;
  executionStatus: string;
  logicalTotal: number;
  logicalApplied: number;
  logicalAlreadySatisfied: number;
  logicalReplaced: number;
  logicalOperatorReview: number;
  logicalForbidden: number;
  logicalPending: number;
  unresolvedRequiredIds: string[];
  unresolvedReasons: Record<string, string>;
  globalBlockers: string[];
  evidenceComplete: boolean;
  stages: Record<string, SoundcheckReadinessStage>;
}

export interface SoundcheckFamilyContract {
  label: string;
  lifecycle: string;
  riskClass: string;
  blockers: string[];
  separateApproval: boolean;
  hardwareProof: string;
  proofRunAllowed: boolean;
  writeKinds: string[];
  liveAllowedWriteKinds: string[];
}

export interface ProposalApplyResult {
  proposalId?: string;
  status: string;
  reason: string;
  applyRequest: string;
  readbackStatus: string;
  rollbackStatus: string;
  rollbackAvailable: boolean;
  readbackTone: Tone;
  rollbackTone: Tone;
  tone: Tone;
  noConsoleSendProven?: boolean | null;
  physicalVerified?: boolean | null;
  physicalStateRestored?: boolean | null;
  writeStateUnknown?: boolean;
  rollbackStateUnknown?: boolean;
  manualReviewRequired?: boolean;
  walTerminal?: string;
  audioRecheckPassed?: boolean | null;
  updatedAt: string;
}

export interface DecisionQueueItem {
  id: string;
  entityKind: 'operator_proposal' | 'mixing_agent_action';
  queueSection: 'pending' | 'history' | 'analysis_only';
  titlePreview: string;
  reasonPreview: string;
  displayTarget: string;
  kind: string;
  severity: string;
  status: string;
  source: string;
  createdAt: string | null;
  updatedAt: string | null;
  expiresAt: string | null;
  canApply: boolean;
  requiresApproval: boolean;
  applyBlockerCodes: string[];
  applyBlockerCount: number;
  linkedChangeCount: number;
  reviewContractVersion: number | null;
  reviewContractStatus: string;
  reviewContentDigest: string | null;
  detailAvailable: boolean;
  hazardScope: string;
  priorityClass: string;
  analysisOnly: boolean;
  importRequired: boolean;
}

export interface DecisionQueueState {
  contractVersion: 2;
  wireSchema: 'automixer.decision_queue.v2';
  availability: 'unavailable' | 'loading' | 'ready' | 'degraded' | 'stale' | 'error' | 'terminal';
  status: string;
  snapshot: {
    id: string;
    revision: string;
    createdAt: string;
    expiresAt: string;
    fullyPageable: boolean;
  } | null;
  page: {
    pageSize: number;
    returnedCount: number;
    remainingCount: number;
    nextCursor: string | null;
    hasMore: boolean;
  };
  counts: {
    totalCount: number;
    includedCount: number;
    omittedCount: number;
    pinnedTotalCount: number;
    pinnedIncludedCount: number;
    pinnedOmittedCount: number;
    bySection: Record<'pending' | 'history' | 'analysis_only', number>;
    byPriority: Record<'global_hazard' | 'blocked' | 'high' | 'ordinary' | 'history', number>;
    byApplyState: Record<'applyable' | 'blocked' | 'analysis_only', number>;
  };
  items: DecisionQueueItem[];
  overflow: {
    countsOnly: boolean;
    reason: string | null;
    unpageableCount: number;
    pinnedUnpageableCount: number;
    applyUiDisabled: boolean;
    hazardDigest: string | null;
  };
  hazards: {
    fatal: boolean;
    applyUiDisabled: boolean;
    count: number;
    codes: string[];
    overflowCount: number;
    sourceCounts: Record<'runtime' | 'manual_write' | 'terminal_result', number>;
    digest: string | null;
  };
  applyUiDisabled: boolean;
  paginationLoading: boolean;
  error: string;
  terminalClose: TransportTerminalClose | null;
}

export interface ProposalReviewState {
  status: 'idle' | 'loading' | 'verified' | 'error' | 'stale';
  proposalId: string | null;
  reviewContractVersion: number | null;
  reviewContentDigest: string | null;
  chunkCount: number;
  receivedChunkCount: number;
  totalRawBytes: number;
  currentState: {
    status: string;
    updated_at: string | null;
    expired: boolean;
    can_apply: boolean;
    apply_blocker_codes: string[];
    apply_blocker_count: number;
  } | null;
  document: Record<string, unknown> | null;
  error: string;
  verifiedAt: string | null;
}

export interface DecisionTraceRecord {
  decisionId: string;
  proposalId?: string;
  source: string;
  mode?: string;
  target?: string;
  actionType: string;
  status: string;
  delta?: string;
  proofStatus: string;
  blockedReason?: string;
  createdAt: string;
  tone: Tone;
}

export interface GoalCapabilityObjectiveReadinessPriority {
  goal_area: string;
  required_row_count?: number;
  ready_row_count?: number;
  gap_row_count?: number;
  readiness_score?: number;
}

export interface GoalCapabilityObjectiveReadinessByGoalArea {
  goal_area: string;
  required_row_count?: number;
  ready_row_count?: number;
  gap_row_count?: number;
  readiness_score?: number;
  blocker_row_count?: number;
  blocker_row_ids?: string[];
  full_autonomous_status?: string;
  full_autonomous_status_label?: string;
  blocker_rows?: string[];
  target_live_writable_row_count?: number;
  target_proof_gap_count?: number;
}

export interface GoalCapabilityObjectiveReadinessHotspot {
  goal_area: string;
  gap_row_count?: number;
  readiness_score?: number;
  blocker_row_count?: number;
  blocker_row_ids?: string[];
}

export interface SafetyMonitorState {
  operatorMode: string;
  liveWritePolicy: string;
  applyPath: string;
  emergencyStopped: boolean;
  resetRequired: boolean;
  ready: boolean;
  mixerConnected: boolean;
  approvedLiveWriteKinds: string[];
  blockedSurfaces: string[];
  lastProofStatus: string;
  lastReadbackStatus: string;
  traceCount: number;
  blockedTraceCount: number;
  sourceLabel: string;
  goalCapabilityObjectiveReadinessScore: number;
  goalCapabilityObjectiveReadinessGoalAreaPriorities: GoalCapabilityObjectiveReadinessPriority[];
  goalCapabilityObjectiveReadinessByGoalArea: Record<string, GoalCapabilityObjectiveReadinessByGoalArea>;
  goalCapabilityObjectiveReadinessHotspots: GoalCapabilityObjectiveReadinessHotspot[];
  proofGoalCapabilityObjectiveReadinessGoalAreaPriorities: GoalCapabilityObjectiveReadinessPriority[];
  proofGoalCapabilityObjectiveReadinessByGoalArea: Record<string, GoalCapabilityObjectiveReadinessByGoalArea>;
  proofGoalCapabilityObjectiveReadinessHotspots: GoalCapabilityObjectiveReadinessHotspot[];
}

export interface ReplayRunRecord {
  runId: string;
  status: string;
  reason: string;
  generatedAt: string;
  reportPath: string;
  tracePath: string;
  metricsPath: string;
  traceCount: number;
  offlineOnly: boolean;
  liveWrite: boolean;
}

export interface ChannelIntelligenceRecord {
  channel: number;
  name: string;
  role: string;
  group: string;
  risk: string;
  gainRideState: string;
  recentTraceCount: number;
  liveApplyAllowed: boolean;
}

export interface ModuleStatusRecord {
  moduleId: string;
  label: string;
  domain: string;
  status: string;
  entrypoint: string;
  module: string;
  uiSurface: string;
  canWriteLive: boolean;
  liveWriteKinds: string[];
  pipelineProfiles: string[];
  blockedReason: string;
  nextStep: string;
  tone: Tone;
}

export interface TelemetryAlertRecord {
  id: string;
  sessionId: string;
  time: string;
  rule: string;
  severity: string;
  target: string;
  message: string;
  value: string;
  threshold: string;
  tone: Tone;
}

export interface TelemetryAlertSessionRecord {
  sessionId: string;
  generatedAt: string;
  eventCount: number;
  alertCount: number;
  dashboardRowCount: number;
  readOnly: boolean;
  liveWrite: boolean;
}

export interface TelemetryAlertsSnapshot {
  available: boolean;
  generatedAt: string;
  sourceLabel: string;
  storageDir: string;
  alertCount: number;
  criticalCount: number;
  warningCount: number;
  latestSessionId: string;
  readOnly: boolean;
  liveWrite: boolean;
  generatesArtifacts: boolean;
  sessions: TelemetryAlertSessionRecord[];
  alerts: TelemetryAlertRecord[];
}

export interface ToastState {
  message: string;
  tone: Tone;
}

export type EmergencyStopStatus = 'ready' | 'pending' | 'acknowledged' | 'error';

export interface EmergencyStopUiState {
  status: EmergencyStopStatus;
  message: string;
  updatedAt: string;
  reconciliationRequired?: boolean;
}

export type FeedbackAlertSeverity = 'critical' | 'warning' | 'info';

export interface FeedbackAlertRecord {
  channel: number;
  frequencyHz: number;
  magnitudeDb: number;
  recommendedAction: string;
  confidence: number;
  autoAction: string;
  readOnly: boolean;
  timestamp: number;
  severity: FeedbackAlertSeverity;
  severityTone: Tone;
  channelLabel: string;
  frequencyLabel: string;
  magnitudeLabel: string;
  confidenceLabel: string;
  actionLabel: string;
  timeLabel: string;
  readOnlyNote: string;
}

export interface ActionFocus {
  action: string;
  target: string;
  detail: string;
  tone: Tone;
  timestamp: string;
}

export type RouteId =
  | 'dashboard'
  | 'soundcheck'
  | 'agents'
  | 'communications'
  | 'task-queue'
  | 'memory'
  | 'performance'
  | 'settings'
  | 'channels'
  | 'groups'
  | 'system'
  | 'routing'
  | 'osc-midi'
  | 'devices'
  | 'logs'
  | 'training'
  | 'rules';

export interface NavItem {
  id: RouteId;
  label: string;
  section: string;
  mark: string;
}

export interface Metric {
  label: string;
  value: string;
  sub?: string;
  tone?: Tone;
}

export interface Channel {
  id: number;
  name: string;
  type: string;
  source: string;
  level: number;
  lufs: number;
  dynamic: number;
  status: 'ACTIVE' | 'MUTED' | 'INACTIVE';
  recognition: string;
  group: string;
  fader: number;
  pan?: number;
  gateLabel?: string;
  compLabel?: string;
  confidence: number;
  meterAvailable?: boolean;
  faderAvailable?: boolean;
  muteAvailable?: boolean;
  panAvailable?: boolean;
  gateAvailable?: boolean;
  compAvailable?: boolean;
  readCoverageStatus?: string;
  cacheFreshnessStatus?: string;
  dataSource?: 'backend' | 'mock';
}

export interface BusRecord {
  name: string;
  role: string;
  level: number;
  lufs: number;
  channels: number;
  color: Tone;
  activeChannels?: number;
  mutedChannels?: number;
  dynamic?: number;
  confidence?: number;
  status?: string;
  members?: string[];
}

export interface Agent {
  id: string;
  name: string;
  role: string;
  status: 'ACTIVE' | 'STANDBY' | 'WARNING' | 'BLOCKED' | 'UNAVAILABLE';
  task: string;
  confidence: number;
  activity: number[];
  lastAction: string;
  tone: Tone;
}

export interface Task {
  id: string;
  priority: 'High' | 'Medium' | 'Low';
  title: string;
  agent: string;
  status: 'Pending' | 'Review' | 'Applied' | 'Rejected';
  time: string;
  target?: string;
  proposalId?: string;
  canApply?: boolean;
  backendStatus?: string;
  reason?: string;
  reviewOnly?: boolean;
  applyBlockers?: string[];
  correctionDomain?: string;
  requestedValueType?: string;
  safetyEvidence?: SafetyEvidenceItem[];
  proposalContract?: ProposalOnlyContract;
  queueSection?: DecisionQueueItem['queueSection'];
  entityKind?: DecisionQueueItem['entityKind'];
  reviewContractVersion?: number | null;
  reviewContractStatus?: string;
  reviewContentDigest?: string | null;
  detailAvailable?: boolean;
  requiresApproval?: boolean;
  hazardScope?: string;
  priorityClass?: string;
  expiresAt?: string | null;
  linkedChangeCount?: number;
}

export interface ProposalOnlyContract {
  valueType: string;
  targetKind: string;
  requiredFields: string[];
  oldValueSource: string;
  safetyEvidenceRequired: string[];
  requiredApproval: string;
  readbackRequired: boolean;
  rollbackRequired: boolean;
  liveApplyAllowed: boolean;
  allowedLiveWriteKinds: string[];
  promotionNext: string;
}

export interface DecisionChainStep {
  node: string;
  state: string;
  tone: Tone;
}

export interface DecisionPolicy {
  actionLabel: string;
  apply: string;
  write: string;
  rollback: string;
  hint: string;
  chain: DecisionChainStep[];
}

export interface ProposalPreview {
  parameter: string;
  delta: string;
  risk: 'Low' | 'Medium' | 'High';
  proof: string;
  rollback: string;
  tone: Tone;
}

export interface SafetyEvidenceItem {
  label: string;
  value: string;
  tone: Tone;
}

export interface ApplyPlanStep {
  label: string;
  value: string;
  tone: Tone;
}

export interface Communication {
  time: string;
  from: string;
  to: string;
  topic: string;
  status: string;
  tone: Tone;
}

export interface MemoryRecord {
  name: string;
  type: string;
  size: string;
  confidence: number;
  updated: string;
}

export interface PerformanceRecord {
  agent: string;
  cpu: string;
  memory: string;
  decisions: number;
  success: string;
  latency: string;
  status: string;
}

export interface DeviceRecord {
  name: string;
  ip: string;
  type: string;
  sampleRate: string;
  io: string;
  status: string;
}

export interface LogRecord {
  time: string;
  level: 'INFO' | 'WARN' | 'ERROR' | 'SUCCESS';
  source: string;
  message: string;
}

export interface GroupsBusesSnapshotDto {
  generatedAt?: string;
  source?: string;
  buses: BusRecord[];
}

export interface MemoryKnowledgeSnapshotDto {
  generatedAt?: string;
  source?: string;
  memories: MemoryRecord[];
}

export interface TrainingSessionSnapshotRow {
  name: string;
  status: string;
  examples: number;
  score: string;
  updated: string;
}

export interface TrainingSessionsSnapshotDto {
  generatedAt?: string;
  source?: string;
  sessions: TrainingSessionSnapshotRow[];
}

export interface RuleLibrarySnapshotRow {
  name: string;
  category: string;
  confidence: string;
  state: string;
}

export interface RulesLibrarySnapshotDto {
  generatedAt?: string;
  source?: string;
  rules: RuleLibrarySnapshotRow[];
}

export interface AgentCommunicationsSnapshotDto {
  generatedAt?: string;
  source?: string;
  communications: Communication[];
}

export interface TopologyNode {
  id: string;
  label: string;
  kind: string;
  status: string;
  detail: string;
}

export interface TopologyLink {
  from: string;
  to: string;
  kind: string;
  status: string;
}

export interface RoutingSegment {
  id: string;
  start: number;
  end: number;
  role: string;
  tapPoint: string;
  labelShort: string;
  labelFull: string;
  wingRoutingHint: string;
  stereo: boolean;
  required: boolean;
  usedBy: string[];
}

export interface DashboardRuntime {
  integratedLufs: number;
  truePeakDb: number;
  shortTermLufs: number;
  momentaryLufs: number;
  lraLu: number;
  aiConfidencePercent: number;
  activeChannels: number;
  totalChannels: number;
  pendingDecisions: number;
  appliedDecisions: number;
  blockedDecisions: number;
  sourceLabel: string;
  masterAvailable: boolean;
  integratedLufsAvailable: boolean;
  truePeakAvailable: boolean;
  shortTermLufsAvailable: boolean;
  momentaryLufsAvailable: boolean;
  lraAvailable: boolean;
  spectrumAvailable: boolean;
  aiConfidenceAvailable: boolean;
  audioCaptureRunning: boolean;
  mainFaderDb?: number;
  mainFaderAvailable: boolean;
  readCoverageStatus: string;
  cacheFreshnessStatus: string;
  spectrumBins?: number[];
  analysisTraces?: Partial<Record<'phase' | 'correlation' | 'dynamics' | 'loudness', DashboardRuntimeTrace>>;
}

export interface DashboardRuntimeTrace {
  values: number[];
  primary?: string;
  secondary?: string;
  sourceLabel?: string;
  sourceTone?: Tone;
}

export interface RuntimeStatusItem {
  label: string;
  value: string;
  tone: Tone;
}

export interface DirectorRuntime {
  currentGoal: string;
  readyLabel: string;
  readyTone: Tone;
  audioLabel: string;
  audioTone: Tone;
  proposalQueueLabel: string;
  proposalQueueTone: Tone;
  liveWritePolicy: string;
  manualGateLabel: string;
  manualGateTone: Tone;
  resetRequired: boolean;
  hardwareProofItems: RuntimeStatusItem[];
  blockedReasons: string[];
  statusItems: RuntimeStatusItem[];
  sourceLabel: string;
}

export interface ViewModel {
  title: string;
  subtitle: string;
  content: ReactNode;
  rightPanel?: ReactNode;
  runtime?: DirectorRuntime;
  mockDataOnly?: boolean;
}

export interface UiState {
  mode: AutomixerMode;
  backend: BackendUiState;
  soundcheck: SoundcheckRuntime;
  emergencyStop: EmergencyStopUiState;
  feedbackAlerts: FeedbackAlertRecord[];
  decisionQueue: DecisionQueueState;
  proposalReview: ProposalReviewState;
  selectedAgentId: string;
  selectedCommunicationKey: string;
  selectedPerformanceAgent: string;
  selectedSettingsSection: string;
  selectedChannelId: number;
  selectedBusName: string;
  selectedTaskId: string;
  selectedTopologyNodeId: string;
  selectedRoutingSegmentId: string;
  selectedOscMessageKey: string;
  selectedDeviceName: string;
  selectedLogKey: string;
  selectedMemoryName: string;
  selectedTrainingName: string;
  selectedRuleName: string;
  supervisedProposalTaskId: string | null;
  taskStatusOverrides: Record<string, Task['status']>;
  actionFocus: ActionFocus | null;
  lastAction: string;
  snapshotCount: number;
  toast: ToastState | null;
}

export interface UiActions {
  setMode: (mode: AutomixerMode) => void;
  selectAgent: (id: string) => void;
  selectCommunication: (key: string) => void;
  selectPerformanceAgent: (agent: string) => void;
  selectSettingsSection: (section: string) => void;
  selectChannel: (id: number) => void;
  selectBus: (name: string) => void;
  selectTask: (id: string) => void;
  selectTopologyNode: (id: string) => void;
  selectRoutingSegment: (id: string) => void;
  selectOscMessage: (key: string) => void;
  selectDevice: (name: string) => void;
  selectLog: (key: string) => void;
  selectMemory: (name: string) => void;
  selectTraining: (name: string) => void;
  selectRule: (name: string) => void;
  openSupervisedProposal: (taskId: string) => void;
  closeSupervisedProposal: () => void;
  confirmSupervisedApply: (taskId: string) => void;
  setTaskStatus: (id: string, status: Task['status']) => void;
  focusAction: (focus: ActionFocus) => void;
  clearActionFocus: () => void;
  createSnapshot: () => void;
  rollbackLastAction: () => void;
  refreshBackend: () => void;
  loadNextDecisionQueuePage: () => void;
  toggleAudioPreview: () => void;
  connectWingReadOnly: () => void;
  navigate: (route: RouteId) => void;
  runBackendAnalysis: () => void;
  startReplayRun: () => void;
  startSoundcheck: () => void;
  stopSoundcheck: () => void;
  acceptSoundcheckReplan: () => void;
  reanalyzeSoundcheckStage: () => void;
  finalizeSoundcheckReadiness: () => void;
  triggerEmergencyStop: () => boolean;
  pushToast: (message: string, tone?: Tone) => void;
  dismissToast: () => void;
}
