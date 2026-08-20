import { AutomixerDashboard } from '../dashboard/AutomixerDashboard';
import type { Channel, DashboardRuntime, InstrumentSpace, UiActions, UiState } from '../../types';

export function DashboardWorkspace({
  channels,
  instrumentSpace,
  runtime,
  ui,
  actions
}: {
  channels: Channel[];
  instrumentSpace: InstrumentSpace;
  runtime: DashboardRuntime;
  ui: UiState;
  actions: UiActions;
}) {
  return <AutomixerDashboard actions={actions} channels={channels} instrumentSpace={instrumentSpace} runtime={runtime} ui={ui} />;
}
