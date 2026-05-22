import { AgentsConsoleRightPanel, AgentsConsoleWorkspace, CommunicationsRightPanel, CommunicationsWorkspace } from '../components/prototype/AgentsConsolePanels';
import type { OperatorViewData } from '../services/useOperatorBackend';
import type { UiActions, UiState } from '../types';
import { tableView } from './viewHelpers';

export function AgentsConsole(ui: UiState, actions: UiActions, data: OperatorViewData) {
  return {
    title: '2. Agents Console',
    subtitle: 'Agent management and AI control',
    rightPanel: <AgentsConsoleRightPanel agents={data.agents} communications={data.communications} selectedAgentId={ui.selectedAgentId} tasks={data.tasks} />,
    content: <AgentsConsoleWorkspace actions={actions} agents={data.agents} metrics={data.overviewMetrics} selectedAgentId={ui.selectedAgentId} />
  };
}

export function Communications(ui: UiState, actions: UiActions, data: OperatorViewData) {
  return tableView('3. Communications', 'Agent message bus and director broadcast', (
    <CommunicationsWorkspace actions={actions} agents={data.agents} communications={data.communications} selectedCommunicationKey={ui.selectedCommunicationKey} source={data.readModelSources.communications} />
  ), <CommunicationsRightPanel agents={data.agents} communications={data.communications} selectedCommunicationKey={ui.selectedCommunicationKey} />);
}
