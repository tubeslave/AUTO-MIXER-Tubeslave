import { DashboardWorkspace } from '../components/prototype/DashboardPanels';
import type { OperatorViewData } from '../services/useOperatorBackend';
import type { UiActions, UiState } from '../types';
import { RightDirectorPanel } from './DirectorPanel';

export function Dashboard(ui: UiState, actions: UiActions, data: OperatorViewData) {
  return {
    title: '1. Overview',
    subtitle: 'Dashboard · Live in Club · Soundcheck',
    runtime: data.directorRuntime,
    rightPanel: <RightDirectorPanel actions={actions} runtime={data.directorRuntime} taskRows={data.tasks} />,
    content: <DashboardWorkspace actions={actions} channels={data.channels} runtime={data.dashboardRuntime} ui={ui} />
  };
}
