import { AutomixerDashboard } from '../dashboard/AutomixerDashboard';
import type { Channel, DashboardRuntime, UiActions, UiState } from '../../types';

export function DashboardWorkspace({
  channels,
  runtime,
  ui,
  actions
}: {
  channels: Channel[];
  runtime: DashboardRuntime;
  ui: UiState;
  actions: UiActions;
}) {
  return <AutomixerDashboard actions={actions} channels={channels} runtime={runtime} ui={ui} />;
}
