import { useCallback, useEffect, useMemo, useState } from 'react';
import { AppShell } from './components/prototype/AutomixerComponents';
import { navItems } from './mock/automixerData';
import { useOperatorBackend } from './services/useOperatorBackend';
import type { ActionFocus, AutomixerMode, RouteId, Task, ToastState, Tone, UiActions, UiState } from './types';
import { getModeDisplayLabel, toneForMode } from './utils/prototypeTone';
import { buildView } from './views/AutomixerViews';

const routeIds = new Set<RouteId>(navItems.map(item => item.id));

function readHashRoute(): RouteId {
  const raw = window.location.hash.replace(/^#\/?/, '') as RouteId;
  return routeIds.has(raw) ? raw : 'dashboard';
}

export default function App() {
  const { backend, backendActions, connected, modeFromBackend, viewData } = useOperatorBackend();
  const [activeRoute, setActiveRoute] = useState<RouteId>(() => readHashRoute());
  const [mode, setMode] = useState<AutomixerMode>('Assist');
  const [selectedAgentId, selectAgent] = useState('analyzer');
  const [selectedCommunicationKey, selectCommunication] = useState('10:30-Analyzer Agent-Decision Agent-Vocal energy exceeds reference by 1.5 dB');
  const [selectedPerformanceAgent, selectPerformanceAgent] = useState('Analyzer Agent');
  const [selectedSettingsSection, selectSettingsSection] = useState('Backend Policy');
  const [selectedChannelId, selectChannel] = useState(12);
  const [selectedBusName, selectBus] = useState('Vocals');
  const [selectedTaskId, selectTask] = useState('T-101');
  const [selectedTopologyNodeId, selectTopologyNode] = useState('automixer_core');
  const [selectedRoutingSegmentId, selectRoutingSegment] = useState('routing-1-2');
  const [selectedOscMessageKey, selectOscMessage] = useState('10:24-System-Connected to WING RACK');
  const [selectedDeviceName, selectDevice] = useState('WING RACK');
  const [selectedLogKey, selectLog] = useState('10:24:15-System-0');
  const [selectedMemoryName, selectMemory] = useState('Vocal intelligibility memory');
  const [selectedTrainingName, selectTraining] = useState('Agent roster calibration');
  const [selectedRuleName, selectRule] = useState('Runtime memory writes stay disabled');
  const [supervisedProposalTaskId, setSupervisedProposalTaskId] = useState<string | null>(null);
  const [taskStatusOverrides, setTaskStatusOverrides] = useState<Record<string, Task['status']>>({});
  const [actionFocus, setActionFocus] = useState<ActionFocus | null>(null);
  const [lastAction, setLastAction] = useState('Last action 10:24:15');
  const [snapshotCount, setSnapshotCount] = useState(0);
  const [toast, setToast] = useState<ToastState | null>(null);

  useEffect(() => {
    const onHashChange = () => setActiveRoute(readHashRoute());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  useEffect(() => {
    if (!toast) return undefined;
    const timer = window.setTimeout(() => setToast(null), 2600);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    if (modeFromBackend && modeFromBackend !== mode) {
      setMode(modeFromBackend);
      setLastAction(`Backend mode ${modeFromBackend}`);
    }
  }, [mode, modeFromBackend]);

  const pushToast = useCallback((message: string, tone: Tone = 'green') => {
    setToast({ message, tone });
  }, []);

  const updateMode = useCallback((nextMode: AutomixerMode) => {
    if (connected) {
      backendActions.setBackendMode(nextMode);
      const displayMode = getModeDisplayLabel(nextMode);
      setLastAction(`Mode request ${displayMode}`);
      pushToast(`Mode request sent to backend: ${displayMode}.`, toneForMode(nextMode));
      return;
    }

    setMode(nextMode);
    const displayMode = getModeDisplayLabel(nextMode);
    setLastAction(`Mode changed to ${displayMode}`);
    pushToast(`Mode changed to ${displayMode}. Mock fallback only.`, toneForMode(nextMode));
  }, [backendActions, connected, pushToast]);

  const setTaskStatus = useCallback((id: string, status: Task['status']) => {
    if (status === 'Applied' && mode === 'Analyze') {
      setActionFocus({
        action: 'Apply Blocked',
        target: id,
        detail: 'Analyze mode is read-only. The proposal stays visible for analysis and no local apply state is recorded.',
        tone: 'red',
        timestamp: '10:24:15'
      });
      const displayMode = getModeDisplayLabel('Analyze');
      setLastAction(`${id} blocked by ${displayMode}`);
      pushToast(`${id} blocked: ${displayMode} is read-only.`, 'red');
      return;
    }

    if (status === 'Applied' && mode === 'Assist') {
      setTaskStatusOverrides(current => ({ ...current, [id]: 'Review' }));
      setActionFocus({
        action: 'Review Queued',
        target: id,
        detail: 'Assist mode can suggest changes but cannot apply them. The proposal was moved into local review state.',
        tone: 'yellow',
        timestamp: '10:24:15'
      });
      const displayMode = getModeDisplayLabel('Assist');
      setLastAction(`${id} queued for review`);
      pushToast(`${id} queued for review. ${displayMode} does not apply changes.`, 'yellow');
      return;
    }

    const task = viewData.tasks.find(item => item.id === id);
    const proposalId = task?.proposalId ?? id;
    if (connected && status === 'Rejected') {
      backendActions.dismissProposal(proposalId);
      setLastAction(`${proposalId} dismiss requested`);
      pushToast(`${proposalId} dismiss request sent to backend.`, 'red');
      return;
    }

    if (connected && status === 'Applied') {
      if (mode !== 'Supervised') {
        setActionFocus({
          action: 'Apply Blocked',
          target: proposalId,
          detail: `Backend apply is only available in ${getModeDisplayLabel('Supervised')} mode. ${getModeDisplayLabel('Analyze')} and ${getModeDisplayLabel('Assist')} cannot send console applies.`,
          tone: 'red',
          timestamp: '10:24:15'
        });
        setLastAction(`${proposalId} blocked by mode`);
        pushToast(`${proposalId} blocked: switch to ${getModeDisplayLabel('Supervised')} for approved apply.`, 'red');
        return;
      }

      if (!task?.canApply) {
        setActionFocus({
          action: 'Apply Blocked',
          target: proposalId,
          detail: 'Backend proposal has no applyable requested_change. It can be reviewed or dismissed only.',
          tone: 'yellow',
          timestamp: '10:24:15'
        });
        setLastAction(`${proposalId} has no applyable change`);
        pushToast(`${proposalId} has no applyable backend change.`, 'yellow');
        return;
      }

      backendActions.applyProposal(proposalId);
      setLastAction(`${proposalId} supervised apply requested`);
      pushToast(`${proposalId} supervised apply request sent to backend.`, 'yellow');
      return;
    }

    setTaskStatusOverrides(current => ({ ...current, [id]: status }));
    const displayMode = getModeDisplayLabel(mode);
    setActionFocus({
      action: status === 'Rejected' ? 'Reject' : 'Apply',
      target: id,
      detail: status === 'Rejected'
        ? 'Task rejected in local prototype state. No mixer command was sent.'
        : mode === 'Auto'
          ? `${displayMode} mode simulated the apply locally. Live writes remain disabled in this prototype.`
          : `${displayMode} mode updated task status locally. No mixer command was sent.`,
      tone: status === 'Rejected' ? 'red' : mode === 'Auto' ? 'yellow' : 'green',
      timestamp: '10:24:15'
    });
    setLastAction(`${id} marked ${status}`);
    pushToast(`${id} marked ${status}. No mixer write sent.`, status === 'Rejected' ? 'red' : mode === 'Auto' ? 'yellow' : 'green');
  }, [backendActions, connected, mode, pushToast, viewData.tasks]);

  const openSupervisedProposal = useCallback((taskId: string) => {
    selectTask(taskId);
    setSupervisedProposalTaskId(taskId);
    setLastAction(`${taskId} supervised proposal opened`);
    pushToast(`${taskId} supervised proposal opened. No mixer write sent.`, 'yellow');
  }, [pushToast]);

  const closeSupervisedProposal = useCallback(() => {
    setSupervisedProposalTaskId(null);
    setLastAction('Supervised proposal closed');
  }, []);

  const confirmSupervisedApply = useCallback((taskId: string) => {
    setTaskStatus(taskId, 'Applied');
    if (!connected) {
      setSupervisedProposalTaskId(null);
    }
  }, [connected, setTaskStatus]);

  const focusAction = useCallback((focus: ActionFocus) => {
    setActionFocus(focus);
    setLastAction(`${focus.action} ${focus.target}`);
    pushToast(`${focus.action} opened for ${focus.target}. Mock UI only.`, focus.tone);
  }, [pushToast]);

  const createSnapshot = useCallback(() => {
    setSnapshotCount(current => {
      const next = current + 1;
      setLastAction(`Snapshot MOCK-${String(next).padStart(3, '0')}`);
      pushToast(`Snapshot MOCK-${String(next).padStart(3, '0')} saved locally.`, 'blue');
      return next;
    });
  }, [pushToast]);

  const rollbackLastAction = useCallback(() => {
    setTaskStatusOverrides({});
    setActionFocus(null);
    setSnapshotCount(0);
    setLastAction('Rollback restored mock state');
    pushToast('Mock UI state rolled back. No console command sent.', 'yellow');
  }, [pushToast]);

  const startReplayRun = useCallback(() => {
    const runId = `frontend-replay-${Date.now()}`;
    if (!connected) {
      backendActions.refreshBackend();
      setActionFocus({
        action: 'Replay Connect',
        target: 'Offline replay',
        detail: 'Backend connection is required before starting an offline replay run. No mixer command was sent.',
        tone: 'yellow',
        timestamp: '10:24:15'
      });
      setLastAction('Replay waiting for backend');
      pushToast('Connect backend before starting offline replay. No live write sent.', 'yellow');
      return;
    }

    backendActions.startReplayRun(runId);
    setActionFocus({
      action: 'Replay Requested',
      target: runId,
      detail: 'Offline replay requested through Product Layer. This is dry-run evidence generation only.',
      tone: 'blue',
      timestamp: '10:24:15'
    });
    setLastAction(`Offline replay ${runId} requested`);
    pushToast('Offline replay requested. Live writes remain disabled.', 'blue');
  }, [backendActions, connected, pushToast]);

  const ui: UiState = useMemo(() => ({
    mode,
    backend,
    selectedAgentId,
    selectedCommunicationKey,
    selectedPerformanceAgent,
    selectedSettingsSection,
    selectedChannelId,
    selectedBusName,
    selectedTaskId,
    selectedTopologyNodeId,
    selectedRoutingSegmentId,
    selectedOscMessageKey,
    selectedDeviceName,
    selectedLogKey,
    selectedMemoryName,
    selectedTrainingName,
    selectedRuleName,
    supervisedProposalTaskId,
    taskStatusOverrides,
    actionFocus,
    lastAction,
    snapshotCount,
    toast
  }), [backend, mode, selectedAgentId, selectedCommunicationKey, selectedPerformanceAgent, selectedSettingsSection, selectedChannelId, selectedBusName, selectedTaskId, selectedTopologyNodeId, selectedRoutingSegmentId, selectedOscMessageKey, selectedDeviceName, selectedLogKey, selectedMemoryName, selectedTrainingName, selectedRuleName, supervisedProposalTaskId, taskStatusOverrides, actionFocus, lastAction, snapshotCount, toast]);

  const actions: UiActions = useMemo(() => ({
    setMode: updateMode,
    selectAgent,
    selectCommunication,
    selectPerformanceAgent,
    selectSettingsSection,
    selectChannel,
    selectBus,
    selectTask,
    selectTopologyNode,
    selectRoutingSegment,
    selectOscMessage,
    selectDevice,
    selectLog,
    selectMemory,
    selectTraining,
    selectRule,
    openSupervisedProposal,
    closeSupervisedProposal,
    confirmSupervisedApply,
    setTaskStatus,
    focusAction,
    clearActionFocus: () => setActionFocus(null),
    createSnapshot,
    rollbackLastAction,
    refreshBackend: backendActions.refreshBackend,
    runBackendAnalysis: backendActions.runBackendAnalysis,
    startReplayRun,
    pushToast,
    dismissToast: () => setToast(null)
  }), [backendActions.refreshBackend, backendActions.runBackendAnalysis, closeSupervisedProposal, confirmSupervisedApply, createSnapshot, focusAction, openSupervisedProposal, pushToast, rollbackLastAction, setTaskStatus, startReplayRun, updateMode]);

  const view = useMemo(() => buildView(activeRoute, ui, actions, viewData), [activeRoute, ui, actions, viewData]);

  const navigate = (route: RouteId) => {
    window.location.hash = `/${route}`;
    setActiveRoute(route);
  };

  return (
    <AppShell
      actions={actions}
      active={activeRoute}
      items={navItems}
      mode={mode}
      onModeChange={updateMode}
      onNavigate={navigate}
      runtime={viewData.directorRuntime}
      ui={ui}
      view={view}
    />
  );
}
