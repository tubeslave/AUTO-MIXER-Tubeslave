import React from 'react';

const PRIMARY_STATE_LABELS = {
  dry_run: 'Dry Run',
  shadow: 'Shadow',
  blocked: 'Blocked',
  confirmed: 'Confirmed',
  rollback_ready: 'Rollback Ready',
  idle: 'Idle',
};

function formatTimestamp(value) {
  if (!value) return 'n/a';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function formatValue(value) {
  if (value === null || value === undefined || value === '') return 'n/a';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(2);
  return String(value);
}

function renderTone(value) {
  return value ? 'ok' : 'warn';
}

export default function PilotDeploymentCockpit({ status, onRefresh }) {
  if (!status) {
    return (
      <div className="connect-card pilot-cockpit-card">
        <div className="pilot-cockpit-header">
          <div>
            <p className="pilot-kicker">Supervised Pilot Deployment</p>
            <h3>Operator Cockpit</h3>
          </div>
          <button className="btn-sm" onClick={onRefresh}>Refresh</button>
        </div>
        <p className="pilot-empty">
          Waiting for backend cockpit status. Connect the backend and mixer, then refresh.
        </p>
      </div>
    );
  }

  const shadow = status.shadow_transport || {};
  const boundary = status.write_boundary || {};
  const supervision = boundary.manual_supervision || {};
  const gate = boundary.write_gate || {};
  const session = status.manual_write_session || {};
  const latestConfirmation = status.latest_confirmation || {};
  const latestBlock = status.latest_block || {};
  const readiness = status.readiness || {};
  const checks = readiness.checks || [];
  const escalations = status.escalations || [];

  return (
    <div className="connect-card pilot-cockpit-card">
      <div className="pilot-cockpit-header">
        <div>
          <p className="pilot-kicker">Supervised Pilot Deployment</p>
          <h3>Operator Cockpit</h3>
          <p className="pilot-meta">
            Updated {formatTimestamp(status.generated_at)} · Mode {formatValue(status.connection_mode)}
          </p>
        </div>
        <div className={`pilot-primary-badge state-${status.primary_state || 'idle'}`}>
          {PRIMARY_STATE_LABELS[status.primary_state] || formatValue(status.primary_state)}
        </div>
        <button className="btn-sm" onClick={onRefresh}>Refresh</button>
      </div>

      <div className="pilot-state-strip">
        {(status.states || []).map((item) => (
          <div
            key={item.key}
            className={`pilot-state-pill ${item.active ? 'active' : 'inactive'} state-${item.key}`}
            title={item.detail}
          >
            <span>{item.label}</span>
            <strong>{item.active ? 'ON' : 'OFF'}</strong>
          </div>
        ))}
      </div>

      <div className="pilot-cockpit-grid">
        <section className="pilot-panel">
          <h4>Write Boundary</h4>
          <dl className="pilot-definition-list">
            <div>
              <dt>Boundary active</dt>
              <dd className={`pilot-status-${renderTone(boundary.deployment_boundary_active)}`}>
                {boundary.deployment_boundary_active ? 'Yes' : 'No'}
              </dd>
            </div>
            <div>
              <dt>Manual path armed</dt>
              <dd>{supervision.armed ? 'Armed' : 'Disarmed'}</dd>
            </div>
            <div>
              <dt>Panic stopped</dt>
              <dd>{supervision.panic_stopped ? 'Yes' : 'No'}</dd>
            </div>
            <div>
              <dt>Gate state</dt>
              <dd>{formatValue(gate.state)}</dd>
            </div>
            <div>
              <dt>Observation only</dt>
              <dd>{boundary.observation_only ? 'Yes' : 'No'}</dd>
            </div>
            <div>
              <dt>Cooldown</dt>
              <dd>{formatValue(supervision.cooldown_sec)} s</dd>
            </div>
          </dl>
          <div className="pilot-limits">
            <span>Fader {formatValue((supervision.limits || {}).fader?.[0])} .. {formatValue((supervision.limits || {}).fader?.[1])} dB</span>
            <span>Gain {formatValue((supervision.limits || {}).gain?.[0])} .. {formatValue((supervision.limits || {}).gain?.[1])} dB</span>
          </div>
        </section>

        <section className="pilot-panel">
          <h4>Confirmation / Readback</h4>
          <dl className="pilot-definition-list">
            <div>
              <dt>Approval required</dt>
              <dd>{supervision.approval_required ? 'Yes' : 'No'}</dd>
            </div>
            <div>
              <dt>Last approval</dt>
              <dd>{formatValue(latestConfirmation.approval_id)}</dd>
            </div>
            <div>
              <dt>Last channel</dt>
              <dd>{formatValue(latestConfirmation.channel)}</dd>
            </div>
            <div>
              <dt>Value type</dt>
              <dd>{formatValue(latestConfirmation.value_type)}</dd>
            </div>
            <div>
              <dt>Effective value</dt>
              <dd>{formatValue(latestConfirmation.effective_value)}</dd>
            </div>
            <div>
              <dt>Readback</dt>
              <dd>{formatValue(latestConfirmation.readback_value)}</dd>
            </div>
          </dl>
          <p className="pilot-detail-line">Confirmed at {formatTimestamp(latestConfirmation.at)}</p>
          {latestBlock.reason && (
            <p className="pilot-detail-line pilot-detail-warn">
              Last block: {latestBlock.reason} at {formatTimestamp(latestBlock.at)}
            </p>
          )}
        </section>

        <section className="pilot-panel">
          <h4>Shadow Transport</h4>
          <dl className="pilot-definition-list">
            <div>
              <dt>State</dt>
              <dd>{formatValue(shadow.state)}</dd>
            </div>
            <div>
              <dt>Dry-run only</dt>
              <dd>{shadow.dry_run_only ? 'Yes' : 'No'}</dd>
            </div>
            <div>
              <dt>Live mixer writes</dt>
              <dd>{shadow.live_mixer_writes ? 'Yes' : 'No'}</dd>
            </div>
            <div>
              <dt>Readiness score</dt>
              <dd>{formatValue(shadow.promotion_readiness_score)}</dd>
            </div>
            <div>
              <dt>Selected proposal</dt>
              <dd>{formatValue(shadow.selected_proposal_id)}</dd>
            </div>
            <div>
              <dt>Disagreements</dt>
              <dd>{formatValue(shadow.disagreement_count)}</dd>
            </div>
          </dl>
          <p className="pilot-detail-line">Report {formatTimestamp(shadow.generated_at)}</p>
          <p className="pilot-detail-line">Dir {formatValue(shadow.report_dir)}</p>
          {(shadow.blocked_reasons || []).length > 0 && (
            <div className="pilot-badge-row">
              {shadow.blocked_reasons.map((reason) => (
                <span key={reason} className="pilot-inline-badge pilot-inline-badge-warn">{reason}</span>
              ))}
            </div>
          )}
        </section>

        <section className="pilot-panel">
          <h4>Rollback / Session</h4>
          <dl className="pilot-definition-list">
            <div>
              <dt>Pilot ready</dt>
              <dd className={`pilot-status-${renderTone(readiness.pilot_ready)}`}>
                {readiness.pilot_ready ? 'Yes' : 'No'}
              </dd>
            </div>
            <div>
              <dt>Session active</dt>
              <dd>{session.active ? 'Yes' : 'No'}</dd>
            </div>
            <div>
              <dt>Session label</dt>
              <dd>{formatValue(session.session_label)}</dd>
            </div>
            <div>
              <dt>Rollback points</dt>
              <dd>{formatValue(session.rollback_ready_count)}</dd>
            </div>
            <div>
              <dt>Last step</dt>
              <dd>{formatValue((session.last_step || {}).value_type)}</dd>
            </div>
            <div>
              <dt>Last step value</dt>
              <dd>{formatValue((session.last_step || {}).effective_value)}</dd>
            </div>
          </dl>
          <p className="pilot-detail-line">Started {formatTimestamp(session.started_at)}</p>
        </section>
      </div>

      <section className="pilot-panel">
        <h4>Readiness Checks</h4>
        <div className="pilot-check-list">
          {checks.map((check) => (
            <div key={check.key} className={`pilot-check-item ${check.ok ? 'ok' : 'warn'}`}>
              <strong>{check.label}</strong>
              <span>{check.ok ? 'OK' : 'Attention'}</span>
              <p>{check.detail}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="pilot-panel">
        <h4>Escalations</h4>
        {escalations.length === 0 ? (
          <p className="pilot-empty">No escalations. Supervision remains within the configured pilot envelope.</p>
        ) : (
          <div className="pilot-escalations">
            {escalations.map((item) => (
              <div key={`${item.code}-${item.message}`} className={`pilot-escalation severity-${item.severity}`}>
                <div className="pilot-escalation-top">
                  <strong>{item.code}</strong>
                  <span>{item.severity}</span>
                </div>
                <p>{item.message}</p>
                <p className="pilot-escalation-action">{item.operator_action}</p>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
