import { TaskDetailsPanel, TaskQueueWorkspace, type TaskRow } from '../components/prototype/TaskQueuePanels';
import { directorBlockedReasonMessages } from '../components/dashboard/directorPanelModel';
import { applyPlanForTask, decisionPolicyForMode, proposalPreviewForTask, safetyEvidenceForTask } from '../mock/automixerData';
import type { OperatorViewData } from '../services/useOperatorBackend';
import type { UiActions, UiState } from '../types';
import { tableView } from './viewHelpers';

export function TaskQueue(ui: UiState, actions: UiActions, data: OperatorViewData) {
  const taskRows: TaskRow[] = data.tasks.map(task => ({ ...task, status: ui.taskStatusOverrides[task.id] ?? task.status, proposal: proposalPreviewForTask(task) }));
  const selectedTask = taskRows.find(task => task.id === ui.selectedTaskId) ?? taskRows[0];
  const policy = decisionPolicyForMode(ui.mode);
  const evidence = safetyEvidenceForTask(selectedTask, ui.mode);
  const applyPlan = applyPlanForTask(selectedTask, ui.mode);
  const integrationBundleApplyBlocked = directorBlockedReasonMessages(data.directorRuntime.blockedReasons).length > 0;
  return tableView('4. Task Queue', 'Prioritized proposals and operator decisions', (
    <TaskQueueWorkspace
      actions={actions}
      applyPlan={applyPlan}
      channelIntelligence={data.channelIntelligence}
      decisionTraces={data.decisionTraces}
      evidence={evidence}
      policy={policy}
      replayRuns={data.replayRuns}
      runtime={data.directorRuntime}
      safetyMonitor={data.safetyMonitor}
      selectedTask={selectedTask}
      taskRows={taskRows}
      ui={ui}
    />
  ), (
    <TaskDetailsPanel
      actions={actions}
      integrationBundleApplyBlocked={integrationBundleApplyBlocked}
      policy={policy}
      task={selectedTask}
      ui={ui}
    />
  ));
}
