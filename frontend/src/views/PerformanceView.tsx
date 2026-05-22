import { DataTable, Panel, Sparkline, StatusBadge } from '../components/prototype/AutomixerComponents';
import { performance as mockPerformance } from '../mock/automixerData';
import type { OperatorViewData } from '../services/useOperatorBackend';
import type { PerformanceRecord, UiActions, UiState } from '../types';
import { tableView, TechnicalChecklist } from './viewHelpers';

function statusToneForPerformance(status: string) {
  const normalized = status.toLowerCase();
  if (normalized.includes('block') || normalized.includes('error')) return 'red';
  if (normalized.includes('watch') || normalized.includes('warn')) return 'yellow';
  return 'green';
}

function PerformanceInspector({ record, sourceLabel }: { record: PerformanceRecord; sourceLabel: string }) {
  return (
    <div className="flex h-full flex-col gap-2">
      <Panel title="Performance Inspector">
        <div className="mb-3 flex items-center justify-between gap-2">
          <h3 className="truncate text-[15px] font-bold text-text">{record.agent}</h3>
          <StatusBadge tone={statusToneForPerformance(record.status)}>{record.status}</StatusBadge>
        </div>
        <TechnicalChecklist rows={[
          { label: 'CPU', value: record.cpu, tone: Number.parseFloat(record.cpu) > 28 ? 'yellow' : 'green' },
          { label: 'Memory', value: record.memory, tone: 'blue' },
          { label: 'Decisions', value: String(record.decisions), tone: 'purple' },
          { label: 'Confidence', value: record.success, tone: Number.parseFloat(record.success) > 90 ? 'green' : 'yellow' },
          { label: 'Latency', value: record.latency, tone: Number.parseFloat(record.latency) > 90 ? 'yellow' : 'green' }
        ]} />
      </Panel>
      <Panel title="Agent Throughput" className="flex-1">
        <div className="h-24"><Sparkline tone="blue" filled values={[25, 48, 38, 70, 55, 86, 58, 92, 64, 76]} /></div>
        <div className="mt-4 h-24"><Sparkline tone="green" filled values={[70, 78, 74, 82, 86, 80, 88, 91, 89, 94]} /></div>
      </Panel>
      <Panel title="Read-only Guard">
        <TechnicalChecklist rows={[
          { label: 'Source', value: sourceLabel, tone: sourceLabel === 'Runtime' ? 'green' : 'blue' },
          { label: 'Detail panel', value: 'Selected row', tone: 'green' },
          { label: 'Live writes', value: 'Disabled', tone: 'red' }
        ]} />
      </Panel>
    </div>
  );
}

export function Performance(ui: UiState, actions: UiActions, data: OperatorViewData) {
  const performanceRows = data.performanceRecords.length ? data.performanceRecords : mockPerformance;
  const selectedRecord = performanceRows.find(row => row.agent === ui.selectedPerformanceAgent) ?? performanceRows[0] ?? mockPerformance[0];
  return tableView('6. Performance', 'Agent runtime and quality metrics', (
    <div className="grid h-full grid-cols-[1.5fr_1fr] gap-2">
      <DataTable
        rows={performanceRows}
        selectedRow={selectedRecord}
        onSelect={row => actions.selectPerformanceAgent(row.agent)}
        rowKey={row => row.agent}
        searchable
        searchPlaceholder="Search agent performance..."
        filters={[
          { label: 'All', predicate: () => true },
          { label: 'Good', predicate: row => row.status === 'Good' },
          { label: 'Watch', predicate: row => row.status !== 'Good' }
        ]}
        getSearchText={row => `${row.agent} ${row.cpu} ${row.memory} ${row.success} ${row.status}`}
        columns={[
          { key: 'agent', label: 'Agent', width: '1.25fr' },
          { key: 'cpu', label: 'CPU', width: '0.6fr' },
          { key: 'memory', label: 'Memory', width: '0.75fr' },
          { key: 'decisions', label: 'Decisions', width: '0.7fr' },
          { key: 'success', label: 'Confidence', width: '0.8fr', render: row => <StatusBadge tone={Number.parseFloat(row.success) > 90 ? 'green' : 'yellow'}>{row.success}</StatusBadge> },
          { key: 'latency', label: 'Latency', width: '0.7fr' },
          { key: 'status', label: 'Status', width: '0.7fr', render: row => <StatusBadge tone={statusToneForPerformance(row.status)}>{row.status}</StatusBadge> }
        ]}
      />
      <div className="grid grid-rows-2 gap-2">
        <Panel title="Decisions per Minute"><Sparkline tone="blue" filled values={[25, 48, 38, 70, 55, 86, 58, 92, 64, 76]} /></Panel>
        <Panel title="Success Rate"><Sparkline tone="green" filled values={[70, 78, 74, 82, 86, 80, 88, 91, 89, 94]} /></Panel>
      </div>
    </div>
  ), <PerformanceInspector record={selectedRecord} sourceLabel={data.performanceRecords.length ? 'Runtime' : 'Mock'} />);
}
