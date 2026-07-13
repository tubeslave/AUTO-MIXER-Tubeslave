import { Panel, StatusBadge } from '../components/prototype/AutomixerComponents';
import type { OperatorViewData } from '../services/useOperatorBackend';
import type { Tone, UiActions, UiState, ViewModel } from '../types';

const stages = [
  ['preflight', 'Console + audio'],
  ['scanning_channels', 'Channel identity'],
  ['reading_state', 'Snapshot'],
  ['waiting_for_signal', 'Line check'],
  ['analyzing', 'Measurement'],
  ['deriving_recommendations', 'Recommendations'],
  ['running', 'Operator review']
] as const;

function stateTone(state: string, error: string): Tone {
  if (error || state === 'error') return 'red';
  if (state === 'running') return 'blue';
  if (state === 'idle' || state === 'stopped') return 'muted';
  return 'blue';
}

function displayState(value: string) {
  return value.replace(/_/g, ' ') || 'idle';
}

export function SoundcheckView(ui: UiState, actions: UiActions, data: OperatorViewData): ViewModel {
  const runtime = ui.soundcheck;
  const readyToStart = ui.backend.connectionState === 'connected'
    && ui.backend.dataState !== 'stale'
    && !ui.backend.staleSnapshot
    && ui.backend.mixerConnected
    && ui.backend.audioCaptureRunning
    && runtime.selectedChannels.length > 0
    && !runtime.running;
  const activeIndex = stages.findIndex(([key]) => key === runtime.state);
  const channels = data.channels.filter(channel => channel.dataSource === 'backend');
  const readiness = runtime.readiness;
  const replan = runtime.replan;
  const canFinalize = ui.backend.connectionState === 'connected'
    && !['', 'idle', 'starting', 'unknown'].includes(readiness.runId);
  const canReanalyze = canFinalize && runtime.running;
  const readinessTone: Tone = readiness.mixReadiness === 'show_mix_ready'
    ? 'green'
    : readiness.mixReadiness === 'safe_core_ready'
      ? 'blue'
      : readiness.safetyState === 'blocked_global'
        ? 'red'
        : 'yellow';

  return {
    title: 'Soundcheck',
    subtitle: 'Measure → derive absolute proposals → operator review. This screen never writes directly to WING.',
    content: (
      <div className="grid h-full min-h-0 grid-rows-[auto_auto_1fr] gap-2">
        <div className="grid grid-cols-6 gap-2">
          {[
            ['Console', ui.backend.mixerConnected ? 'Connected' : 'Required', ui.backend.mixerConnected ? 'green' : 'red', 'readback'],
            ['Audio', ui.backend.audioCaptureRunning ? `${ui.backend.audioCaptureChannels} ch live` : 'Required', ui.backend.audioCaptureRunning ? 'green' : 'red', 'capture'],
            ['Pipeline', runtime.proposalOnly ? 'Proposal only' : 'Blocked mode', runtime.proposalOnly ? 'blue' : 'red', 'zero write'],
            ['Recommendations', String(runtime.recommendationCount), runtime.derivationComplete ? 'blue' : 'yellow', 'derived only'],
            ['Generated plan', `${runtime.transactionPlanCount} / ${runtime.transactionOperationCount}`, runtime.transactionPlanCount ? 'blue' : 'yellow', 'children / ops'],
            ['Mix readiness', readiness.mixReadiness.replace(/_/g, ' '), readinessTone, readiness.executionStatus.replace(/_/g, ' ')]
          ].map(([label, value, tone, badge]) => (
            <Panel key={label} title={label}>
              <div className="flex items-end justify-between gap-2">
                <strong className="text-[18px] text-text">{value}</strong>
                <StatusBadge tone={tone as Tone}>{badge}</StatusBadge>
              </div>
            </Panel>
          ))}
        </div>

        <Panel title="Soundcheck transport">
          <div className="flex flex-wrap items-center gap-2">
            <button
              className="am-button"
              disabled={!readyToStart}
              onClick={actions.startSoundcheck}
              type="button"
            >
              Start proposal-only soundcheck
            </button>
            <button
              className="am-button"
              disabled={!runtime.running}
              onClick={actions.stopSoundcheck}
              type="button"
            >
              Stop
            </button>
            <button className="am-button" onClick={actions.refreshBackend} type="button">Refresh status</button>
            <button
              className="am-button"
              disabled={!replan.draftAvailable || ui.backend.connectionState !== 'connected'}
              onClick={actions.acceptSoundcheckReplan}
              type="button"
            >
              Accept safe next revision
            </button>
            <button
              className="am-button"
              disabled={!canReanalyze}
              onClick={actions.reanalyzeSoundcheckStage}
              type="button"
            >
              Re-analyze stage (read only)
            </button>
            <button
              className="am-button"
              disabled={!canFinalize}
              onClick={actions.finalizeSoundcheckReadiness}
              type="button"
            >
              Reconcile targets + Main (read only)
            </button>
            <StatusBadge tone={stateTone(runtime.state, runtime.error)}>{displayState(runtime.state)}</StatusBadge>
            <span className="text-[10px] text-muted">{runtime.error || runtime.message}</span>
          </div>
          {!readyToStart && !runtime.running && (
            <p className="mt-2 text-[10px] text-warn">
              Start unlocks only after Product Layer, mixer readback and multichannel audio capture are online.
            </p>
          )}
        </Panel>

        <div className="grid min-h-0 grid-cols-[minmax(0,1fr)_310px] gap-2">
          <Panel className="min-h-0 overflow-auto" title="Real channel evidence">
            <div className="mb-3 grid grid-cols-7 gap-1">
              {stages.map(([key, label], index) => {
                const passed = activeIndex >= 0 && index < activeIndex;
                const active = index === activeIndex;
                return (
                  <div
                    className={`rounded border px-2 py-2 text-center text-[9px] ${active ? 'border-audio bg-audio/10 text-audio' : passed ? 'border-online/40 text-online' : 'border-line text-muted'}`}
                    key={key}
                  >
                    <div className="font-bold">{index + 1}</div>
                    <div>{label}</div>
                  </div>
                );
              })}
            </div>
            {channels.length ? (
              <div className="grid grid-cols-2 gap-2 xl:grid-cols-3">
                {channels.map(channel => (
                  <div className="rounded border border-line bg-panel2 p-2" key={channel.id}>
                    <div className="flex items-center justify-between gap-2">
                      <strong className="truncate text-[11px] text-text">{channel.id}. {channel.name}</strong>
                      <StatusBadge tone={channel.meterAvailable ? 'green' : 'yellow'}>
                        {channel.meterAvailable ? 'signal data' : 'no meter'}
                      </StatusBadge>
                    </div>
                    <div className="mt-2 grid grid-cols-3 gap-2 font-mono text-[10px] text-muted">
                      <span>RMS {channel.meterAvailable ? `${channel.level.toFixed(1)} dB` : '—'}</span>
                      <span>LUFS {channel.meterAvailable ? channel.lufs.toFixed(1) : '—'}</span>
                      <span>FDR {channel.faderAvailable ? `${channel.fader.toFixed(1)} dB` : '—'}</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="grid h-40 place-items-center rounded border border-dashed border-line text-[11px] text-muted">
                No backend channel inventory. Mock channels are intentionally hidden from soundcheck.
              </div>
            )}
          </Panel>

          <div className="flex min-h-0 flex-col gap-2">
            <Panel title="Safety contract">
              <div className="space-y-2 text-[10px]">
                {[
                  ['Direct mixer writes', 'Transaction gate only', 'green'],
                  ['Existing processing', 'Preserved', 'green'],
                  ['Input boost', 'Blocked', 'red'],
                  ['Apply path', 'Operator queue only', 'yellow'],
                  ['M005 acceptance', 'External review required', 'yellow'],
                  ['Standing family authority', `${runtime.liveAllowedFamilies.length} live`, runtime.liveAllowedFamilies.length ? 'yellow' : 'red']
                ].map(([label, value, tone]) => (
                  <div className="flex items-center justify-between gap-2" key={label}>
                    <span className="text-muted">{label}</span>
                    <StatusBadge tone={tone as Tone}>{value}</StatusBadge>
                  </div>
                ))}
              </div>
            </Panel>
            <Panel title="Console module authority">
              <div className="max-h-40 space-y-1 overflow-auto text-[10px]">
                {Object.entries(runtime.familyLifecycle).length ? Object.entries(runtime.familyLifecycle).map(([familyId, lifecycle]) => {
                  const contract = runtime.familyContracts[familyId];
                  const effectiveLiveKinds = (runtime.familyLiveWriteKinds[familyId] || [])
                    .filter(kind => ui.backend.writeScope.includes(kind));
                  const standingLive = lifecycle === 'live_allowed'
                    && effectiveLiveKinds.length > 0
                    && ui.backend.canApplyToConsole
                    && ui.backend.connectionState === 'connected'
                    && !ui.backend.staleSnapshot;
                  return (
                    <div className="flex items-center justify-between gap-2" key={familyId}>
                      <span className="min-w-0 truncate text-muted" title={(contract?.blockers || []).join(', ')}>
                        {familyId.replace(/_/g, ' ')}
                        {(contract?.blockers || []).length > 0 && (
                          <small className="ml-1 text-warn">· {contract.blockers[0].replace(/_/g, ' ')}</small>
                        )}
                      </span>
                      <StatusBadge tone={standingLive ? 'green' : lifecycle === 'hardware_proven' ? 'yellow' : 'red'}>
                        {standingLive
                          ? `${effectiveLiveKinds.length} live`
                          : lifecycle === 'hardware_proven'
                            ? 'hardware proven · owner pending'
                            : 'proposal only'}
                      </StatusBadge>
                    </div>
                  );
                }) : (
                  <p className="text-muted">Waiting for backend family registry.</p>
                )}
              </div>
            </Panel>
            <Panel className="flex-1" title="Evidence summary">
              <div className="space-y-3 text-[10px] text-muted">
                <p><strong className="text-text">Channels with signal:</strong> {runtime.channelsWithSignal} / {runtime.totalChannels || channels.length}</p>
                <p><strong className="text-text">Derivation:</strong> {runtime.derivationComplete ? 'complete' : 'waiting'}</p>
                <p><strong className="text-text">Mix readiness:</strong> {readiness.mixReadiness.replace(/_/g, ' ')}</p>
                <p><strong className="text-text">Required unresolved:</strong> {readiness.unresolvedRequiredIds.length}; operator review: {readiness.logicalOperatorReview}; forbidden: {readiness.logicalForbidden}</p>
                <p><strong className="text-text">Replan:</strong> {replan.draftAvailable ? `revision ${replan.fromPlanRevision} → ${replan.toPlanRevision}; ${replan.splitOperationCount} split; ${replan.safeAlternativeOperationCount} safe alternatives` : replan.status.replace(/_/g, ' ')}</p>
                {replan.safeAlternatives.length > 0 && (
                  <div className="rounded border border-audio/30 bg-audio/5 p-2 text-audio">
                    <strong className="text-text">Trusted safe alternative:</strong>
                    <ul className="mt-1 list-disc space-y-1 pl-4">
                      {replan.safeAlternatives.map((alternative, index) => (
                        <li key={`${alternative.strategy}-${index}`}>
                          {alternative.strategy.replace(/_/g, ' ')}
                          {typeof alternative.translationDb === 'number' ? ` (${alternative.translationDb.toFixed(1)} dB)` : ''}
                          {alternative.finalTargetVectorPreserved ? ' · final target vector preserved' : ''}
                          {alternative.relativeBalancePreserved ? ' · current step preserves relative balance' : ''}
                          {alternative.currentStepTransitional ? ' · transitional bounded step' : ''}
                          {alternative.constructiveLaneUsed ? ' · constructive' : ' · down-only'}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {replan.operatorRequests.length > 0 && (
                  <p><strong className="text-text">Replan review:</strong> {replan.operatorRequests.filter(request => !request.safeAutomaticReplacementAvailable).length} corrections still require a separate operator decision.</p>
                )}
                <p><strong className="text-text">Safety state:</strong> {readiness.safetyState.replace(/_/g, ' ')}</p>
                {readiness.globalBlockers.length > 0 && (
                  <div className="rounded border border-danger/30 bg-danger/5 p-2 text-danger">
                    <strong className="text-text">Global blockers:</strong>
                    <ul className="mt-1 list-disc space-y-1 pl-4">
                      {readiness.globalBlockers.slice(0, 8).map(reason => (
                        <li key={reason}>{reason.replace(/_/g, ' ')}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {Object.entries(readiness.unresolvedReasons).length > 0 && (
                  <div className="rounded border border-warn/30 bg-warn/5 p-2 text-warn">
                    <strong className="text-text">Why incomplete:</strong>
                    <ul className="mt-1 list-disc space-y-1 pl-4">
                      {Object.entries(readiness.unresolvedReasons).slice(0, 8).map(([operationId, reason]) => (
                        <li key={operationId} title={operationId}>{reason.replace(/_/g, ' ')}</li>
                      ))}
                    </ul>
                  </div>
                )}
                <p><strong className="text-text">Intercepted writes:</strong> {runtime.interceptedWriteCount} (evidence only; sent = 0)</p>
                <p><strong className="text-text">Generated plans:</strong> {runtime.transactionPlanCount}; semantic operations: {runtime.transactionOperationCount}</p>
                <p><strong className="text-text">Updated:</strong> {runtime.lastUpdated || 'not received'}</p>
                <p className="rounded border border-warn/30 bg-warn/5 p-2 text-warn">
                  USB Main L/R is console mix evidence, not room/PA truth. Final room validation stays with the operator and measurement microphone.
                </p>
              </div>
            </Panel>
          </div>
        </div>
      </div>
    ),
    runtime: data.directorRuntime,
    mockDataOnly: false
  };
}
