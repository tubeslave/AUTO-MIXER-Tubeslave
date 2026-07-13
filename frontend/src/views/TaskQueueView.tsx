import { TaskDetailsPanel, TaskQueueWorkspace, type TaskRow } from '../components/prototype/TaskQueuePanels';
import { directorBlockedReasonMessages } from '../components/dashboard/directorPanelModel';
import { applyPlanForTask, decisionPolicyForMode, proposalPreviewForTask, safetyEvidenceForTask } from '../mock/automixerData';
import type { OperatorViewData } from '../services/useOperatorBackend';
import type { UiActions, UiState } from '../types';
import { tableView } from './viewHelpers';

export function TaskQueue(ui: UiState, actions: UiActions, data: OperatorViewData) {
  const taskRows: TaskRow[] = data.tasks.map(task => ({ ...task, status: ui.taskStatusOverrides[task.id] ?? task.status, proposal: proposalPreviewForTask(task) }));
  if (taskRows.length === 0) {
    const queue = data.decisionQueue;
    const unavailable = ['unavailable', 'error', 'stale', 'terminal'].includes(queue.availability);
    return tableView('4. Task Queue', 'Prioritized proposals and operator decisions', (
      <div className="grid h-full place-items-center p-5">
        <div className="w-full max-w-[720px] rounded-[8px] border border-line bg-row p-5 text-[11px] leading-5 text-muted">
          <b className="block text-base text-text">
            {queue.availability === 'ready'
              ? 'Decision Queue v2 is genuinely empty'
              : 'Decision Queue v2 is unavailable'}
          </b>
          <p className="mt-2">
            {queue.availability === 'ready'
              ? 'The backend returned items: []. No demo proposals were inserted.'
              : `${queue.error || queue.availability}. Apply remains blocked and stale/demo rows are not substituted.`}
          </p>
          <p className="mt-2 font-mono text-[10px]">
            total={queue.counts.totalCount} included={queue.counts.includedCount} omitted={queue.counts.omittedCount}
          </p>
          {(queue.page.hasMore || unavailable) && (
            <div className="mt-4 flex gap-2">
              {queue.page.hasMore && (
                <button className="am-button" onClick={actions.loadNextDecisionQueuePage} type="button">
                  Load next page
                </button>
              )}
              {unavailable && (
                <button className="am-button" onClick={actions.refreshBackend} type="button">
                  {queue.availability === 'terminal' ? 'Manual retry' : 'Refresh queue'}
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    ));
  }
  const selectedTask = taskRows.find(task => task.id === ui.selectedTaskId) ?? taskRows[0];
  const policy = decisionPolicyForMode(ui.mode);
  const evidence = safetyEvidenceForTask(selectedTask, ui.mode);
  const applyPlan = applyPlanForTask(selectedTask, ui.mode);
  const queueApplyBlocked = data.decisionQueue.availability !== 'ready'
    || data.decisionQueue.applyUiDisabled;
  const integrationBundleApplyBlocked = directorBlockedReasonMessages(data.directorRuntime.blockedReasons).length > 0
    || queueApplyBlocked;
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
