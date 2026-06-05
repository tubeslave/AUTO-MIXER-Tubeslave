import { expect, test } from '@playwright/test';

const routes = [
  ['dashboard', '1. Overview'],
  ['agents', '2. Agents Console'],
  ['communications', '3. Communications'],
  ['task-queue', '4. Task Queue'],
  ['memory', '5. Memory / Knowledge Base'],
  ['performance', '6. Performance'],
  ['settings', '7. Settings'],
  ['channels', '8. Channel List'],
  ['groups', '9. Groups / Buses'],
  ['system', '10. System Connection'],
  ['routing', '11. Audio Routing'],
  ['osc-midi', '12. OSC / MIDI'],
  ['devices', '13. Devices'],
  ['logs', '14. Logs'],
  ['training', '15. Training Sessions'],
  ['rules', '16. Rules Library']
] as const;

test.describe('Automixer frontend prototype', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem('automixer_backend_autoconnect', 'off');
    });
  });

  test('renders every reference route without framework errors', async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('console', message => {
      if (message.type() === 'error') consoleErrors.push(message.text());
    });
    page.on('pageerror', error => consoleErrors.push(error.message));

    for (const [route, title] of routes) {
      await page.goto(`/#/${route}`);
      await expect(page.getByText(title)).toBeVisible();
      await expect(page.locator('body')).toContainText('Automixer');
    }

    expect(consoleErrors).toEqual([]);
  });

  test('keeps operator actions local and does not emit unsafe writes', async ({ page }) => {
    const unsafeRequests: string[] = [];
    page.on('request', request => {
      if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(request.method())) {
        unsafeRequests.push(`${request.method()} ${request.url()}`);
      }
    });

    await page.goto('/#/task-queue');
    await expect(page.getByText('4. Task Queue')).toBeVisible();
    await expect(page.getByText('Decision Chain')).toBeVisible();
    await expect(page.getByText('Proposal Preview')).toBeVisible();
    await expect(page.getByText('Safety Evidence')).toBeVisible();
    await expect(page.getByText('Apply Plan')).toBeVisible();
    await page.getByLabel('Automation mode').selectOption('Supervised');
    await expect(page.getByText('Mode changed to Supervised. Mock fallback only.')).toBeVisible();

    const leadVocalRow = page.locator('.am-table-row').filter({ hasText: 'Reduce Lead Vocal presence' });
    await leadVocalRow.getByRole('button', { name: 'Apply' }).click();

    await expect(page.getByRole('dialog', { name: 'Supervised Apply Proposal' })).toBeVisible();
    await expect(page.getByRole('dialog', { name: 'Supervised Apply Proposal' })).toContainText('Current console snapshot');
    await expect(page.getByRole('dialog', { name: 'Supervised Apply Proposal' })).toContainText('Snapshot required');
    await page.getByRole('button', { name: 'Request Supervised Apply' }).click();
    await expect(page.getByText('T-101 marked Applied. No mixer write sent.')).toBeVisible();
    await expect(leadVocalRow).toContainText('Applied');

    await page.goto('/#/channels');
    const kickInRow = page.locator('.am-table-row').filter({ hasText: 'Kick In' });
    await kickInRow.getByRole('button', { name: 'Inspect' }).click();

    const actionFocus = page.locator('section').filter({ hasText: 'Action Focus' }).filter({ hasText: 'Kick In' });
    await expect(actionFocus).toBeVisible();
    await expect(actionFocus.getByText('Write Path')).toBeVisible();
    await expect(actionFocus.getByText('Guarded')).toBeVisible();
    expect(unsafeRequests).toEqual([]);
  });

  test('mode policy gates task apply actions locally', async ({ page }) => {
    await page.goto('/#/task-queue');
    await expect(page.getByText('4. Task Queue')).toBeVisible();
    await expect(page.locator('section').filter({ hasText: 'Mode Guard' })).toContainText('Assist');

    const leadVocalRow = page.locator('.am-table-row').filter({ hasText: 'Reduce Lead Vocal presence' });

    await page.getByLabel('Automation mode').selectOption('Analyze');
    await expect(page.locator('section').filter({ hasText: 'Mode Guard' })).toContainText('Analyze');
    await leadVocalRow.getByRole('button', { name: 'Apply' }).click();
    const blockedFocus = page.locator('section').filter({ hasText: 'Action Focus' }).filter({ hasText: 'Apply Blocked' });
    await expect(blockedFocus).toContainText('Analyze mode is read-only');
    await expect(leadVocalRow).toContainText('Pending');

    await page.getByLabel('Automation mode').selectOption('Assist');
    await expect(page.locator('section').filter({ hasText: 'Mode Guard' })).toContainText('Assist');
    await leadVocalRow.getByRole('button', { name: 'Apply' }).click();
    await expect(page.getByText('Assist mode does not apply changes')).toBeVisible();
    await expect(leadVocalRow).toContainText('Review');

    await page.getByLabel('Automation mode').selectOption('Auto');
    await expect(page.locator('section').filter({ hasText: 'Mode Guard' })).toContainText('Auto');
    await leadVocalRow.getByRole('button', { name: 'Apply' }).click();
    await expect(page.getByText('Auto mode simulated the apply locally')).toBeVisible();
    await expect(page.getByRole('dialog', { name: 'Supervised Apply Proposal' })).toHaveCount(0);
  });

  test('mode selector changes prototype state only', async ({ page }) => {
    await page.goto('/#/dashboard');
    await page.getByLabel('Automation mode').selectOption('Supervised');

    await expect(page.getByText('Mode changed to Supervised. Mock fallback only.')).toBeVisible();
    await expect(page.getByLabel('Automation mode')).toHaveValue('Supervised');
  });

  test('renders compact operator view on mobile width', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/#/dashboard');

    await expect(page.getByText('Compact Operator')).toBeVisible();
    await expect(page.getByText('Selected Channel')).toBeVisible();
    await expect(page.getByLabel('Compact mode')).toBeVisible();
  });

  test('selected rows update read-only detail panels', async ({ page }) => {
    await page.goto('/#/communications');
    await page.locator('.am-table-row').filter({ hasText: 'Review Snare transient' }).click();
    await expect(page.locator('section').filter({ hasText: 'Selected Message' })).toContainText('Review Snare transient');

    await page.goto('/#/memory');
    await page.locator('.am-table-row').filter({ hasText: 'Bass DI compression memory' }).click();
    await expect(page.locator('section').filter({ hasText: 'Memory Inspector' })).toContainText('Bass DI compression memory');
    await expect(page.locator('section').filter({ hasText: 'Memory Inspector' })).toContainText('Dynamics');

    await page.goto('/#/groups');
    await page.locator('.am-table-row').filter({ hasText: 'Bass Bus' }).click();
    await expect(page.locator('section').filter({ hasText: 'Bus Details' })).toContainText('Bass Bus');

    await page.goto('/#/training');
    await page.locator('.am-table-row').filter({ hasText: 'Rock drums transient map' }).click();
    const trainingDetail = page.locator('section').filter({ hasText: 'Training Queue' }).filter({ hasText: 'Auto learn' });
    await expect(trainingDetail).toContainText('Rock drums transient map');
    await expect(trainingDetail).toContainText('84 examples');

    await page.goto('/#/rules');
    await page.locator('.am-table-row').filter({ hasText: 'No boost without headroom' }).click();
    await expect(page.locator('section').filter({ hasText: 'Rule Inspector' })).toContainText('No boost without headroom');
    await expect(page.locator('section').filter({ hasText: 'Rule Inspector' })).toContainText('Locked');

    await page.goto('/#/performance');
    await page.locator('.am-table-row').filter({ hasText: 'Learning Agent' }).click();
    await expect(page.locator('section').filter({ hasText: 'Performance Inspector' })).toContainText('Learning Agent');
    await expect(page.locator('section').filter({ hasText: 'Performance Inspector' })).toContainText('Watch');

    await page.goto('/#/settings');
    await page.getByRole('button', { name: 'Write Policy' }).first().click();
    await expect(page.locator('section').filter({ hasText: 'Settings Inspector' })).toContainText('Write Policy');
    await expect(page.locator('section').filter({ hasText: 'Settings Inspector' })).toContainText('Direct OSC');

    await page.goto('/#/system');
    await page.getByRole('button', { name: /WING Rack/ }).click();
    await expect(page.locator('section').filter({ hasText: 'Connection Status' })).toContainText('WING Rack');

    await page.goto('/#/routing');
    await page.getByRole('button', { name: /Bass Bus/ }).click();
    await expect(page.locator('section').filter({ hasText: 'Routing Segment' })).toContainText('Bass Bus');

    await page.goto('/#/osc-midi');
    await page.locator('.am-table-row').filter({ hasText: 'OSC connection established' }).click();
    await expect(page.locator('section').filter({ hasText: 'OSC / MIDI Message' }).first()).toContainText('OSC connection established');

    await page.goto('/#/devices');
    await page.locator('.am-table-row').filter({ hasText: 'Stage Box 1' }).click();
    await expect(page.locator('section').filter({ hasText: 'Selected Device' })).toContainText('Stage Box 1');

    await page.goto('/#/logs');
    await page.locator('.am-table-row').filter({ hasText: 'Redundant link not available' }).click();
    await expect(page.locator('section').filter({ hasText: 'Log Details' })).toContainText('Redundant link not available');
  });
});

test.describe('Automixer backend Product Layer adapter', () => {
  test('shows partial backend state when only some Product Layer payloads arrive', async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem('automixer_backend_autoconnect', 'on');
      const payloads: Record<string, unknown> = {
        get_operator_mode: {
          type: 'operator_mode_status',
          mode: 'assist',
          label: 'Assist',
          live_write_policy: 'read_only_partial_fixture',
          apply_path: 'none',
          capabilities: {
            can_create_proposals: true,
            can_apply_to_console: false,
            requires_approval: true,
            allowed_live_write_kinds: []
          }
        },
        get_dashboard_snapshot: {
          type: 'dashboard_snapshot',
          generated_at: '2026-05-19T20:40:00Z',
          operator_mode: {
            mode: 'assist',
            label: 'Assist',
            live_write_policy: 'read_only_partial_fixture',
            apply_path: 'none',
            capabilities: {
              can_create_proposals: true,
              can_apply_to_console: false,
              requires_approval: true,
              allowed_live_write_kinds: []
            }
          },
          connection: {
            connected: false,
            connection_mode: 'wing',
            client_type: 'EnhancedOSCClient',
            ip: '10.10.10.20'
          },
          master_bus: {
            available: true,
            integrated_lufs: -20.1,
            short_term_lufs: -19.8,
            momentary_lufs: -19.1,
            true_peak_db: -3.4,
            lra_lu: 4.8
          },
          channel_summary: {
            total_channels: 48,
            active_channels: 0
          },
          decision_summary: {
            pending_count: 0,
            history_count: 0,
            applied_count: 0,
            blocked_count: 0
          },
          safety: {
            live_write_policy: 'read_only_partial_fixture',
            ready_for_live: {
              ready: false,
              checks: {
                mixer_connected: false,
                gain_staging_available: false,
                auto_fader_available: false,
                no_clipping: true
              }
            },
            manual_write_supervision: {
              emergency_stopped: false,
              requires_operator_mode: 'supervised'
            },
            blocked_reasons: ['Partial Product Layer fixture']
          }
        }
      };

      class PartialWebSocket {
        static CONNECTING = 0;
        static OPEN = 1;
        static CLOSING = 2;
        static CLOSED = 3;
        readyState = PartialWebSocket.CONNECTING;
        onopen: null | (() => void) = null;
        onmessage: null | ((event: { data: string }) => void) = null;
        onerror: null | ((event: unknown) => void) = null;
        onclose: null | (() => void) = null;

        constructor() {
          setTimeout(() => {
            this.readyState = PartialWebSocket.OPEN;
            this.onopen?.();
          }, 0);
        }

        send(raw: string) {
          const message = JSON.parse(raw);
          const response = payloads[message.type];
          if (response) {
            setTimeout(() => this.onmessage?.({ data: JSON.stringify(response) }), 0);
          }
        }

        close() {
          this.readyState = PartialWebSocket.CLOSED;
          this.onclose?.();
        }
      }

      Object.defineProperty(window, 'WebSocket', {
        configurable: true,
        value: PartialWebSocket
      });
    });

    await page.goto('/#/settings');
    await expect(page.locator('section').filter({ hasText: 'Settings Inspector' })).toContainText('Product Layer partial');
    await expect(page.locator('body')).toContainText('read_only_partial_fixture');
    await expect(page.locator('body')).toContainText('No apply');
  });

  test('keeps explicit empty backend DTO snapshots empty instead of injecting mock rows', async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem('automixer_backend_autoconnect', 'on');
      const payloads: Record<string, unknown> = {
        get_operator_mode: {
          type: 'operator_mode_status',
          mode: 'assist',
          live_write_policy: 'read_only_empty_fixture',
          apply_path: 'none',
          capabilities: {
            can_create_proposals: false,
            can_apply_to_console: false,
            requires_approval: true,
            allowed_live_write_kinds: []
          }
        },
        get_dashboard_snapshot: {
          type: 'dashboard_snapshot',
          generated_at: '2026-05-30T15:00:00Z',
          operator_mode: {
            mode: 'assist',
            live_write_policy: 'read_only_empty_fixture',
            apply_path: 'none',
            capabilities: {
              can_create_proposals: false,
              can_apply_to_console: false,
              requires_approval: true,
              allowed_live_write_kinds: []
            }
          },
          connection: { connected: false, connection_mode: 'wing', ip: '10.10.10.20' },
          master_bus: {},
          channel_summary: { total_channels: 0, active_channels: 0 },
          decision_summary: { pending_count: 0, history_count: 0, applied_count: 0, blocked_count: 0 },
          safety: {
            live_write_policy: 'read_only_empty_fixture',
            ready_for_live: { ready: false, checks: {} },
            manual_write_supervision: { emergency_stopped: false }
          }
        },
        get_groups_buses_snapshot: {
          type: 'groups_buses_snapshot',
          source: 'backend_empty_groups',
          buses: []
        },
        get_memory_knowledge_snapshot: {
          type: 'memory_knowledge_snapshot',
          source: 'backend_empty_memory',
          memories: []
        },
        get_training_sessions_snapshot: {
          type: 'training_sessions_snapshot',
          source: 'backend_empty_training',
          sessions: []
        },
        get_rules_library_snapshot: {
          type: 'rules_library_snapshot',
          source: 'backend_empty_rules',
          rules: []
        },
        get_agent_communications_snapshot: {
          type: 'agent_communications_snapshot',
          source: 'backend_empty_agent_bus',
          communications: []
        },
        get_connection_topology: {
          type: 'connection_topology',
          source: 'backend_empty_topology',
          connected_devices: [],
          nodes: [],
          links: [],
          routing_scheme: []
        }
      };

      class EmptySnapshotWebSocket {
        static CONNECTING = 0;
        static OPEN = 1;
        static CLOSING = 2;
        static CLOSED = 3;
        readyState = EmptySnapshotWebSocket.CONNECTING;
        onopen: null | (() => void) = null;
        onmessage: null | ((event: { data: string }) => void) = null;
        onerror: null | ((event: unknown) => void) = null;
        onclose: null | (() => void) = null;

        constructor() {
          setTimeout(() => {
            this.readyState = EmptySnapshotWebSocket.OPEN;
            this.onopen?.();
          }, 0);
        }

        send(raw: string) {
          const message = JSON.parse(raw);
          const response = payloads[message.type];
          if (response) {
            setTimeout(() => this.onmessage?.({ data: JSON.stringify(response) }), 0);
          }
        }

        close() {
          this.readyState = EmptySnapshotWebSocket.CLOSED;
          this.onclose?.();
        }
      }

      Object.defineProperty(window, 'WebSocket', {
        configurable: true,
        value: EmptySnapshotWebSocket
      });
    });

    await page.goto('/#/groups');
    await expect(page.getByText('No group/bus rows received').first()).toBeVisible();
    await expect(page.getByText('Bass Bus')).toHaveCount(0);

    await page.goto('/#/memory');
    await expect(page.getByText('No memory rows received').first()).toBeVisible();
    await expect(page.getByText('Bass DI compression memory')).toHaveCount(0);

    await page.goto('/#/training');
    await expect(page.getByText('No training sessions received').first()).toBeVisible();
    await expect(page.getByText('Rock drums transient map')).toHaveCount(0);

    await page.goto('/#/rules');
    await expect(page.getByText('No rules received').first()).toBeVisible();
    await expect(page.getByText('No boost without headroom')).toHaveCount(0);

    await page.goto('/#/communications');
    await expect(page.getByText('No rows to display.').first()).toBeVisible();
    await expect(page.getByText('Backend DTO').first()).toBeVisible();

    await page.goto('/#/system');
    await expect(page.getByText('No topology nodes received').first()).toBeVisible();
    await expect(page.getByText('WING Rack')).toHaveCount(0);

    await page.goto('/#/routing');
    await expect(page.getByText('No routing segment selected').first()).toBeVisible();
    await expect(page.getByText('Bass Bus')).toHaveCount(0);

    await page.goto('/#/devices');
    await expect(page.getByText('No devices received').first()).toBeVisible();
    await expect(page.getByText('Stage Box 1')).toHaveCount(0);
  });

  test('renders backend payloads through the prototype UI without unsafe writes', async ({ page }) => {
    const sentTypes: string[] = [];

    await page.addInitScript(() => {
      window.localStorage.setItem('automixer_backend_autoconnect', 'on');
      const payloads: Record<string, unknown> = {
        get_operator_mode: {
          type: 'operator_mode_status',
          mode: 'supervised',
          label: 'Supervised',
          live_write_policy: 'supervised_approval_required',
          apply_path: 'supervised_manual_gate',
          capabilities: {
            can_create_proposals: true,
            can_apply_to_console: true,
            requires_approval: true,
            allowed_live_write_kinds: ['fader', 'gain']
          }
        },
        get_dashboard_snapshot: {
          type: 'dashboard_snapshot',
          generated_at: '2026-05-19T20:30:00Z',
          operator_mode: {
            mode: 'supervised',
            label: 'Supervised',
            live_write_policy: 'supervised_approval_required',
            apply_path: 'supervised_manual_gate',
            capabilities: {
              can_create_proposals: true,
              can_apply_to_console: true,
              requires_approval: true,
              allowed_live_write_kinds: ['fader', 'gain']
            }
          },
          connection: {
            connected: true,
            connection_mode: 'wing',
            client_type: 'EnhancedOSCClient',
            ip: '10.10.10.20'
          },
          master_bus: {
            available: true,
            integrated_lufs: -18.2,
            short_term_lufs: -17.8,
            momentary_lufs: -16.9,
            true_peak_db: -2.4,
            lra_lu: 5.1
          },
          audio_capture: {
            running: true,
            num_channels: 48,
            sample_rate: 48000
          },
          channel_summary: {
            total_channels: 2,
            active_channels: 1
          },
          decision_summary: {
            pending_count: 1,
            history_count: 2,
            applied_count: 1,
            blocked_count: 0
          },
          safety: {
            live_write_policy: 'supervised_approval_required',
            ready_for_live: {
              ready: true,
              checks: {
                mixer_connected: true,
                gain_staging_available: true,
                auto_fader_available: true,
                no_clipping: true
              }
            },
            manual_write_supervision: {
              emergency_stopped: false,
              requires_operator_mode: 'supervised'
            },
            blocked_reasons: []
          },
          runtime: {
            agent_available: true,
            proposal_queue_available: true
          }
        },
        get_channel_inventory: {
          type: 'channel_inventory',
          available: true,
          connection: {
            connected: true,
            connection_mode: 'wing',
            client_type: 'EnhancedOSCClient',
            ip: '10.10.10.20'
          },
          channels: [
            {
              id: 1,
              number: 1,
              name: 'Backend Kick',
              source: 'In 01',
              role: 'Kick',
              group: 'DRUMS',
              status: 'active',
              level_db: -13.4,
              lufs: -16.1,
              dynamic_range_db: 7.2,
              control_state: { fader_db: -4.5 },
              ai_recognition: { label: 'Kick', confidence: 98 }
            },
            {
              id: 2,
              number: 2,
              name: 'Backend Lead Vox',
              source: 'In 02',
              role: 'Vocal',
              group: 'VOCALS',
              status: 'idle',
              level_db: -44,
              lufs: -50,
              dynamic_range_db: 3.1,
              control_state: { fader_db: -8 },
              ai_recognition: { label: 'Lead Vocal', confidence: 91 }
            }
          ]
        },
        get_decision_queue: {
          type: 'decision_queue',
          status: 'ok',
          available: true,
          operator_mode: {
            mode: 'supervised',
            label: 'Supervised',
            live_write_policy: 'supervised_approval_required',
            apply_path: 'supervised_manual_gate',
            capabilities: {
              can_create_proposals: true,
              can_apply_to_console: true,
              requires_approval: true,
              allowed_live_write_kinds: ['fader', 'gain']
            }
          },
          summary: {
            pending_count: 1,
            history_count: 2,
            applied_count: 1,
            blocked_count: 0
          },
          pending_actions: [
            {
              id: 'proposal-backend-1',
              title: 'Backend SafeGain trim move',
              target: 'Backend Lead Vox',
              kind: 'gain',
              severity: 'high',
              status: 'pending',
              source: 'Backend Decision Agent',
              reason: 'SafeGain measured input headroom and suggests an input trim adjustment. Ensemble context inactive (ensemble_time_context_stale); time stale; sources 1/2; lag 10/2 samples; operator review only.',
              created_at: '2026-05-19T20:29:00Z',
              requested_change: {
                value_type: 'gain',
                channel: 2,
                value: -9.5,
                safety_evidence: [
                  {
                    label: 'Ensemble context',
                    value: 'ensemble_time_context_stale',
                    reason: 'ensemble_time_context_stale'
                  },
                  {
                    label: 'Ensemble time coherence',
                    value: 'stale',
                    reason: 'ensemble_time_context_stale'
                  },
                  {
                    label: 'Ensemble time sources',
                    value: 1,
                    target: 2
                  },
                  {
                    label: 'Ensemble lag samples',
                    value: 10,
                    target: 2
                  },
                  {
                    label: 'Ensemble apply policy',
                    value: 'operator_review_only'
                  },
                  {
                    label: 'Distortion risk',
                    value: 'low'
                  }
                ]
              },
              can_apply: true
            }
          ],
          history: []
        },
        get_decision_traces: {
          type: 'decision_traces',
          status: 'ok',
          source: 'backend_fixture_trace_store',
          summary: {
            trace_count: 1,
            applied_count: 1,
            blocked_count: 0,
            verified_count: 1
          },
          traces: [
            {
              decision_id: 'trace-fixture-1',
              proposal_id: 'proposal-backend-1',
              source: 'Backend Decision Agent',
              mode: 'supervised',
              target: 2,
              action_type: 'gain',
              status: 'applied',
              current_value: -9,
              proposed_value: -9.5,
              delta: -0.5,
              proof_status: 'verified',
              created_at: '2026-05-19T20:29:05Z'
            }
          ]
        },
        get_safety_monitor: {
          type: 'safety_monitor',
          status: 'ok',
          source: 'backend_fixture_safety',
          operator_mode: {
            mode: 'supervised',
            live_write_policy: 'supervised_approval_required',
            apply_path: 'supervised_manual_gate',
            capabilities: {
              can_apply_to_console: true,
              allowed_live_write_kinds: ['fader', 'gain']
            }
          },
          manual_write_supervision: {
            emergency_stopped: false,
            reset_required: false,
            approved_live_write_kinds: ['fader', 'gain']
          },
          live_write_policy: {
            policy: 'supervised_fader_gain_only',
            apply_path: 'supervised_manual_gate',
            approved_live_write_kinds: ['fader', 'gain'],
            direct_writes_allowed: false,
            autonomous_apply_allowed: false
          },
          proof: {
            last_proof_status: 'verified',
            last_readback: { status: 'verified_from_cache' }
          },
          blocked_surfaces: ['eq', 'compressor', 'routing', 'snapshots', 'autonomous_apply'],
          summary: {
            ready: false,
            mixer_connected: true,
            emergency_stopped: false,
            reset_required: false,
            trace_count: 1,
            blocked_trace_count: 0
          }
        },
        get_replay_runs: {
          type: 'replay_runs',
          status: 'ok',
          source: 'backend_fixture_replay',
          runs: [
            {
              run_id: 'fixture-replay-1',
              status: 'passed',
              reason: 'offline_simulated_apply_readback_rollback_complete',
              generated_at: '2026-05-19T20:28:00Z',
              trace_count: 7,
              offline_only: true,
              live_write: false,
              report_path: 'artifacts/replay/fixture-replay-1/report.md',
              decision_traces_path: 'artifacts/replay/fixture-replay-1/decision_traces.jsonl',
              metrics_path: 'artifacts/replay/fixture-replay-1/metrics_before_after.json'
            }
          ]
        },
        start_replay_run: {
          type: 'replay_run_started',
          status: 'passed',
          success: true,
          offline_only: true,
          live_write: false,
          run_id: 'fixture-replay-started',
          artifacts: {
            report_path: 'artifacts/replay/fixture-replay-started/report.md',
            decision_traces_path: 'artifacts/replay/fixture-replay-started/decision_traces.jsonl',
            metrics_path: 'artifacts/replay/fixture-replay-started/metrics_before_after.json'
          },
          result: {
            status: 'passed',
            write_policy: 'offline_only_no_live_writes',
            summary: {
              trace_count: 3
            }
          }
        },
        get_channel_intelligence: {
          type: 'channel_intelligence',
          status: 'ok',
          source: 'backend_fixture_channel_intelligence',
          channels: [
            {
              channel: 2,
              name: 'Backend Lead Vox',
              role: 'Vocal',
              group: 'VOCALS',
              risk: 'low',
              gain_ride_state: 'shadow_proposal_only',
              recent_trace_count: 1,
              live_apply_allowed: false
            }
          ]
        },
        get_operator_agent_roster: {
          type: 'operator_agent_roster',
          status: 'ok',
          summary: {
            agent_count: 2,
            active_count: 1,
            blocked_count: 0,
            can_live_execute: true
          },
          agents: [
            {
              id: 'automixer_director',
              name: 'Backend Director',
              role: 'Coordinates backend agents',
              status: 'ACTIVE',
              task: 'Supervised backend mode',
              confidence: 96,
              tone: 'green',
              category: 'director'
            },
            {
              id: 'console_executor_agent',
              name: 'Backend Executor',
              role: 'Supervised apply gate',
              status: 'ACTIVE',
              task: 'Ready for approved fader/gain apply',
              confidence: 100,
              tone: 'green',
              category: 'execution',
              can_write: true,
              write_scope: ['fader', 'gain']
            }
          ]
        },
        get_connection_topology: {
          type: 'connection_topology',
          connection: {
            connected: true,
            connection_mode: 'wing',
            client_type: 'EnhancedOSCClient',
            ip: '10.10.10.20'
          },
          connected_devices: [
            {
              name: 'Backend WING',
              ip: '10.10.10.20',
              kind: 'wing',
              sample_rate: '48 kHz',
              detail: 'Backend fixture console',
              status: 'online'
            }
          ],
          nodes: [
            {
              id: 'director_control',
              label: 'Backend Director Surface',
              kind: 'client',
              status: 'online',
              detail: 'fake websocket client'
            },
            {
              id: 'automixer_core',
              label: 'Backend Core Node',
              kind: 'core',
              status: 'running',
              detail: 'mode=supervised'
            },
            {
              id: 'mixer',
              label: 'Backend Mixer Node',
              kind: 'mixer',
              status: 'online',
              detail: 'wing'
            }
          ],
          links: [
            {
              from: 'director_control',
              to: 'automixer_core',
              kind: 'websocket',
              status: 'online'
            },
            {
              from: 'automixer_core',
              to: 'mixer',
              kind: 'osc_control',
              status: 'online'
            }
          ],
          routing_scheme: [
            {
              start: 1,
              end: 2,
              role: 'backend_fixture',
              tap_point: 'Backend Tap Point',
              label_short: 'Backend Routing Segment',
              label_full: 'Backend fixture routing segment',
              wing_routing_hint: 'Backend WING routing hint',
              stereo: true,
              required: true,
              used_by: ['Backend Test Module']
            }
          ]
        },
        get_groups_buses_snapshot: {
          type: 'groups_buses_snapshot',
          generated_at: '2026-05-19T20:30:03Z',
          source: 'backend_fixture_groups',
          buses: [
            {
              name: 'Backend DTO Drums',
              role: 'Group',
              level: -13.4,
              lufs: -16.1,
              channels: 1,
              active_channels: 1,
              muted_channels: 0,
              dynamic: 7.2,
              confidence: 98,
              status: 'Active',
              color: 'green',
              members: ['01 Backend Kick']
            },
            {
              name: 'Backend DTO Vocals',
              role: 'Group',
              level: -44,
              lufs: -50,
              channels: 1,
              active_channels: 0,
              muted_channels: 0,
              dynamic: 3.1,
              confidence: 91,
              status: 'Inactive',
              color: 'purple',
              members: ['02 Backend Lead Vox']
            }
          ]
        },
        get_memory_knowledge_snapshot: {
          type: 'memory_knowledge_snapshot',
          generated_at: '2026-05-19T20:30:04Z',
          source: 'backend_fixture_memory',
          memories: [
            {
              name: 'Backend DTO vocal memory',
              type: 'DTO Memory',
              size: '3.1 KB',
              confidence: 97,
              updated: 'backend fixture'
            },
            {
              name: 'Backend DTO decision memory',
              type: 'Decision History',
              size: '2.0 KB',
              confidence: 92,
              updated: 'queue'
            }
          ]
        },
        get_training_sessions_snapshot: {
          type: 'training_sessions_snapshot',
          generated_at: '2026-05-19T20:30:05Z',
          source: 'backend_fixture_training',
          sessions: [
            {
              name: 'Backend DTO training set',
              status: 'Review',
              examples: 42,
              score: '91%',
              updated: 'backend fixture'
            }
          ]
        },
        get_rules_library_snapshot: {
          type: 'rules_library_snapshot',
          generated_at: '2026-05-19T20:30:06Z',
          source: 'backend_fixture_rules',
          rules: [
            {
              name: 'Backend DTO no direct writes',
              category: 'Safety',
              confidence: '100%',
              state: 'Locked'
            },
            {
              name: 'Backend DTO supervised gate only',
              category: 'Control',
              confidence: '100%',
              state: 'Active'
            }
          ]
        },
        get_agent_communications_snapshot: {
          type: 'agent_communications_snapshot',
          generated_at: '2026-05-19T20:30:07Z',
          source: 'backend_fixture_agent_bus',
          communications: [
            {
              time: '2026-05-19T20:30:07Z',
              from: 'Backend DTO Director',
              to: 'Operator',
              topic: 'Backend DTO communication online',
              status: 'Info',
              tone: 'blue'
            }
          ]
        },
        get_processing_pipeline_snapshot: {
          type: 'processing_pipeline_snapshot',
          generated_at: '2026-05-19T20:30:07Z',
          source: 'backend_fixture_processing',
          main_pipeline_id: 'main_live_assist',
          main_pipeline_status: 'canonical',
          live_apply_policy: 'supervised_fader_gain_only',
          module_status_map: [
            {
              module_id: 'product_operator_state',
              label: 'Backend Product State',
              domain: 'product_layer',
              status: 'canonical',
              entrypoint: 'get_dashboard_snapshot / get_decision_queue / get_safety_monitor',
              module: 'backend/operator_product_state.py',
              ui_surface: 'Control Center dashboard, Task Queue, Safety Monitor',
              can_write_live: false,
              live_write_kinds: [],
              pipeline_profiles: ['main_live_assist'],
              blocked_reason: '',
              next_step: 'Keep as canonical read model.'
            },
            {
              module_id: 'supervised_manual_fader_gain',
              label: 'Backend Supervised Fader/Gain',
              domain: 'gain_fader',
              status: 'canonical',
              entrypoint: 'apply_operator_proposal -> supervised manual gate',
              module: 'backend/server.py',
              ui_surface: 'Task Queue supervised apply',
              can_write_live: true,
              live_write_kinds: ['fader', 'gain'],
              pipeline_profiles: ['main_live_assist'],
              blocked_reason: '',
              next_step: 'Keep proof evidence narrow.'
            },
            {
              module_id: 'automixer_production_mix_v1',
              label: 'Backend Production Mix v1',
              domain: 'offline_mix',
              status: 'deprecated',
              entrypoint: 'none in this checkout',
              module: 'automixer.production_mix_v1',
              ui_surface: 'Hidden from live cockpit',
              can_write_live: false,
              live_write_kinds: [],
              pipeline_profiles: ['quarantine'],
              blocked_reason: 'legacy entrypoint unavailable',
              next_step: 'Restore as offline entrypoint or retire.'
            }
          ],
          supervised_proof_evidence: {
            type: 'supervised_proof_evidence_snapshot',
            available: true,
            report_path: 'artifacts/no_wing_release_check_current_cycle/latest/supervised_proof_artifact_report.json',
            safety: {
              read_only: true,
              live_write: false,
              generates_artifacts: false,
              live_widening_allowed: false,
              automatic_safety_update_allowed: false,
              full_autonomous_ready: false
            },
            summary: {
              latest_report_status: 'ok',
              latest_report_age_sec: 180,
              latest_report_stale: false,
              stale_after_sec: 86400,
              artifact_count: 1,
              invalid_summary_count: 0,
              overlay_status: 'dry_run_only',
              overlay_row_count: 2,
              read_surface_proof_quality: {
                available: true,
                status: 'fixture_evidence_incomplete',
                surface_count: 10,
                surface_with_fixture_count: 10,
                declared_fixture_path_count: 10,
                declared_test_name_count: 25,
                physical_gap_surface_count: 10,
                summary_recomputed_from_rows: true,
                fixture_declaration_validation: true,
                contract_gap_consistency_enforced: true,
                full_autonomous_ready: false,
                physical_fixture_missing_surface_ids: [
                  'raw_mixer_state_cache',
                  'channel_identity_and_basic_controls',
                  'channel_eq_dynamics_gate',
                  'bus_group_main_dca_state',
                  'matrix_state',
                  'fx_slots_and_inserts',
                  'routing_and_sends',
                  'snapshots_scenes',
                  'user_presets',
                  'dlive_state'
                ],
                next_physical_readback_surface_id: 'raw_mixer_state_cache',
                source_next_physical_readback_surface_id: 'raw_mixer_state_cache',
                next_physical_readback_target_consistent: true,
                raw_mixer_state_physical_readback_contract: {
                  available: true,
                  status: 'passed',
                  schema_id: 'raw_mixer_state_physical_readback.v1',
                  contract_status: 'contract_ready_physical_fixture_missing',
                  surface_id: 'raw_mixer_state_cache',
                  remediation_action_id: 'remediate_raw_mixer_state_cache',
                  artifact_required: true,
                  physical_fixture_gap_remains: true,
                  accepted_as_physical_fixture_candidate: false,
                  live_authority_granted: false,
                  full_autonomous_ready: false,
                  next_physical_readback_surface_id: 'raw_mixer_state_cache',
                  next_physical_readback_action_id: 'remediate_raw_mixer_state_cache',
                  target_consistency: true,
                  allowed_sources: ['physical_readback', 'wing_physical_readback'],
                  accepted_row_statuses: ['verified_from_physical_readback'],
                  forbidden_mutation_evidence_fields: ['manual_write_events'],
                  violations: []
                }
              }
            }
          },
          read_surface_remediation_plan: {
            type: 'read_surface_remediation_plan',
            status: 'read_remediation_required',
            safety: {
              read_only: true,
              live_write: false,
              mutates_console_state: false,
              live_widening_allowed: false,
              full_autonomous_ready: false
            },
            summary: {
              surface_count: 10,
              incomplete_surface_count: 10,
              action_count: 10,
              physical_fixture_missing_surface_ids: [
                'raw_mixer_state_cache',
                'channel_identity_and_basic_controls'
              ],
              next_physical_readback_surface_id: 'raw_mixer_state_cache',
              next_physical_readback_action_id: 'remediate_raw_mixer_state_cache',
              global_blockers: [
                'full_console_read_coverage_incomplete',
                'physical_readback_fixtures_incomplete',
                'read_surface_remediation_does_not_contact_mixer'
              ],
              next_global_step: 'Complete read DTOs and physical fixtures before broad autonomous console control.'
            }
          },
          autonomy_evidence_overlay: {
            rows: [
              {
                id: 'fader_autonomy_evidence_overlay',
                target_id: 'fader_supervised_promotion',
                value_type: 'fader',
                status: 'no_current_evidence',
                physical_candidate_count: 0,
                dry_run_ready_count: 0,
                blocked_artifact_count: 0,
                remediation_action_count: 0,
                physical_candidate_proof_ids: [],
                dry_run_ready_proof_ids: [],
                blocked_proof_ids: [],
                remediation_action_ids: [],
                blockers: ['proof_artifact_missing'],
                next_required_actions: ['Run one physical supervised fader proof.']
              },
              {
                id: 'gain_autonomy_evidence_overlay',
                target_id: 'gain_trim_supervised_promotion',
                value_type: 'gain',
                status: 'dry_run_only',
                physical_candidate_count: 0,
                dry_run_ready_count: 1,
                blocked_artifact_count: 0,
                remediation_action_count: 0,
                physical_candidate_proof_ids: [],
                dry_run_ready_proof_ids: ['gain-proof-dry-run'],
                blocked_proof_ids: [],
                remediation_action_ids: [],
                blockers: ['physical_proof_required_for_promotion'],
                next_required_actions: ['Run one small supervised gain proof on WING.']
              }
            ]
          }
        },
        get_main_pipeline_preview: {
          type: 'main_pipeline_preview',
          generated_at: '2026-05-19T20:30:08Z',
          source: 'backend_fixture_pipeline_preview',
          status: 'ok',
          pipeline_id: 'main_live_assist',
          pipeline_status: 'canonical',
          executor_policy: {
            preview_only: true,
            queue_mutation: false,
            live_write: false,
            allowed_requested_change_value_types: ['fader', 'gain']
          },
          summary: {
            proposal_count: 1,
            proposal_only_contract_count: 2
          },
          proposals: [
            {
              id: 'preview-backend-1',
              title: 'Backend preview fader move',
              requested_change: {
                value_type: 'fader',
                channel: 2,
                value: -9.5
              }
            }
          ],
          proposal_only_contracts: [
            {
              id: 'main_live_assist:proposal_only_contract:auto_eq',
              adapter_id: 'proposal_only_contracts',
              method_id: 'auto_eq',
              method_label: 'Backend Auto EQ',
              domain: 'eq',
              status: 'proposal_only',
              title: 'Backend Auto EQ: proposal contract only',
              reason: 'EQ recommendations are review-only until hardware readback and rollback are proven.',
              risk: 'medium',
              target: 'channel_eq_band',
              queue_action: 'not_queueable_until_promoted',
              can_import_to_queue: false,
              can_apply_live: false,
              blockers: [
                'unsupported_value_type:eq',
                'not_in_supervised_fader_gain_executor',
                'hardware_readback_and_rollback_not_proven'
              ],
              proposal_contract: {
                value_type: 'eq',
                target_kind: 'channel_eq_band',
                required_fields: ['target', 'old_value', 'proposed_value', 'safety_evidence', 'rollback_value'],
                old_value_source: 'cached_or_physical_eq_band_read_required_before_promotion',
                safety_evidence_required: ['frequency_match', 'feedback_notch_preserved'],
                required_approval: 'operator_supervised',
                readback_required: true,
                rollback_required: true,
                live_apply_allowed: false,
                allowed_live_write_kinds: ['fader', 'gain'],
                promotion_next: 'proposal_adapter_no_direct_set_eq'
              }
            },
            {
              id: 'main_live_assist:proposal_only_contract:auto_compressor',
              adapter_id: 'proposal_only_contracts',
              method_id: 'auto_compressor',
              method_label: 'Backend Auto Compressor',
              domain: 'dynamics',
              status: 'proposal_only',
              title: 'Backend Auto Compressor: proposal contract only',
              reason: 'Compressor recommendations are review-only until supervised dynamics proof exists.',
              risk: 'medium',
              target: 'channel_dynamics',
              queue_action: 'not_queueable_until_promoted',
              can_import_to_queue: false,
              can_apply_live: false,
              blockers: ['unsupported_value_type:compressor', 'not_in_supervised_fader_gain_executor'],
              proposal_contract: {
                value_type: 'compressor',
                target_kind: 'channel_dynamics',
                required_fields: ['target', 'old_value', 'proposed_value', 'safety_evidence', 'rollback_value'],
                old_value_source: 'cached_or_physical_dynamics_read_required_before_promotion',
                safety_evidence_required: ['gain_reduction_estimate', 'true_peak_headroom'],
                required_approval: 'operator_supervised',
                readback_required: true,
                rollback_required: true,
                live_apply_allowed: false,
                allowed_live_write_kinds: ['fader', 'gain'],
                promotion_next: 'proposal_adapter_and_hardware_proof_required'
              }
            }
          ],
          blockers: []
        },
        apply_operator_proposal: {
          type: 'operator_proposal_apply_result',
          status: 'applied',
          success: true,
          proposal: {
            id: 'proposal-backend-1',
            title: 'Backend SafeGain trim move'
          },
          apply_result: {
            status: 'applied',
            channel: 2,
            value_type: 'gain',
            effective_value: -9.5,
            readback_status: 'verified_from_cache',
            rollback_available: true,
            rollback_value: -8
          }
        }
      };

      class FakeWebSocket {
        static CONNECTING = 0;
        static OPEN = 1;
        static CLOSING = 2;
        static CLOSED = 3;
        readyState = FakeWebSocket.CONNECTING;
        onopen: null | (() => void) = null;
        onmessage: null | ((event: { data: string }) => void) = null;
        onerror: null | ((event: unknown) => void) = null;
        onclose: null | (() => void) = null;
        url: string;

        constructor(url: string) {
          this.url = url;
          setTimeout(() => {
            this.readyState = FakeWebSocket.OPEN;
            this.onopen?.();
          }, 0);
        }

        send(raw: string) {
          const message = JSON.parse(raw);
          window.dispatchEvent(new CustomEvent('automixer-ws-send', { detail: message.type }));
          const response = payloads[message.type];
          if (response) {
            setTimeout(() => this.onmessage?.({ data: JSON.stringify(response) }), 0);
          }
        }

        close() {
          this.readyState = FakeWebSocket.CLOSED;
          this.onclose?.();
        }
      }

      Object.defineProperty(window, 'WebSocket', {
        configurable: true,
        value: FakeWebSocket
      });
    });

    await page.exposeFunction('recordWsSend', (type: string) => sentTypes.push(type));
    await page.addInitScript(() => {
      window.addEventListener('automixer-ws-send', event => {
        const detail = (event as CustomEvent<string>).detail;
        void (window as unknown as { recordWsSend: (type: string) => Promise<void> }).recordWsSend(detail);
      });
    });

    await page.goto('/#/dashboard');
    await expect(page.locator('header').filter({ hasText: 'Status' }).getByText(/Partial data/i)).toBeVisible();
    await expect(page.getByRole('banner').getByText(/Product Layer partial/i)).toBeVisible();
    await expect(page.getByText('WING 10.10.10.20')).toBeVisible();
    await expect(page.getByText('-18.2').first()).toBeVisible();
    await expect(page.getByText('-2.4').first()).toBeVisible();
    await expect(page.locator('aside').getByText('Ready for live').first()).toBeVisible();
    await expect(page.locator('aside').getByText('Audio capture running').first()).toBeVisible();
    await expect(page.getByText('Backend Kick')).toBeVisible();

    await page.goto('/#/task-queue');
    await expect(page.getByText('Backend SafeGain trim move').first()).toBeVisible();
    await page.locator('.am-table-row').filter({ hasText: 'Backend SafeGain trim move' }).click();
    await expect(page.locator('aside').filter({ hasText: 'Task Details' })).toContainText('Backend Reason');
    await expect(page.locator('aside').filter({ hasText: 'Task Details' })).toContainText('Ensemble context inactive');
    await expect(page.locator('section').filter({ hasText: 'Safety Evidence' }).first()).toContainText('Ensemble context');
    await expect(page.locator('section').filter({ hasText: 'Safety Evidence' }).first()).toContainText('stale');
    await expect(page.locator('section').filter({ hasText: 'Safety Evidence' }).first()).toContainText('1/2');
    await expect(page.getByText('Backend Auto EQ: proposal contract only').first()).toBeVisible();
    const reviewOnlyRow = page.locator('.am-table-row').filter({ hasText: 'Backend Auto EQ: proposal contract only' });
    await expect(reviewOnlyRow.getByRole('button', { name: 'Apply' })).toHaveCount(0);
    await reviewOnlyRow.getByRole('button', { name: 'Detail' }).click();
    await expect(page.locator('aside').getByText('Apply Blockers').first()).toBeVisible();
    await expect(page.locator('aside').getByText('unsupported value type:eq').first()).toBeVisible();
    await expect(page.getByText('Backend Decision Agent').first()).toBeVisible();
    await expect(page.getByText('Backend Lead Vox').first()).toBeVisible();
    await expect(page.getByText('Replay Engine / Evidence').first()).toBeVisible();
    await expect(page.getByText('fixture-replay-1').first()).toBeVisible();
    await expect(page.getByText('no live write').first()).toBeVisible();
    await expect(page.getByText('shadow_proposal_only').first()).toBeVisible();
    await page.getByRole('button', { name: 'Start Offline' }).click();
    await expect(page.getByText('Offline replay requested. Live writes remain disabled.').first()).toBeVisible();
    await expect(page.getByText('fixture-replay-started').first()).toBeVisible();
    await expect(page.getByText('3 rows').first()).toBeVisible();
    await page.locator('.am-table-row').filter({ hasText: 'Backend SafeGain trim move' }).getByRole('button', { name: 'Apply' }).click();
    await expect(page.getByRole('dialog', { name: 'Supervised Apply Proposal' })).toContainText('Backend Lead Vox');
    await expect(page.getByRole('dialog', { name: 'Supervised Apply Proposal' })).toContainText('Backend supervised gate');
    await page.getByRole('button', { name: 'Request Supervised Apply' }).click();
    await expect(page.getByRole('dialog', { name: 'Supervised Apply Proposal' })).toContainText('Backend Result');
    await expect(page.getByRole('dialog', { name: 'Supervised Apply Proposal' })).toContainText('verified_from_cache');
    await expect(page.getByRole('dialog', { name: 'Supervised Apply Proposal' })).toContainText('rollback available');
    await page.getByRole('button', { name: 'Cancel' }).click();

    await page.goto('/#/agents');
    await expect(page.getByText('Backend Director').first()).toBeVisible();
    await expect(page.getByText('Backend Executor').first()).toBeVisible();

    await page.goto('/#/communications');
    await expect(page.getByText('Backend DTO Director').first()).toBeVisible();
    await expect(page.getByText('Backend DTO communication online').first()).toBeVisible();
    await expect(page.getByText('Backend DTO').first()).toBeVisible();

    await page.goto('/#/memory');
    await expect(page.getByText('Backend DTO vocal memory').first()).toBeVisible();
    await expect(page.getByText('Backend DTO decision memory').first()).toBeVisible();
    await expect(page.getByText('Backend DTO').first()).toBeVisible();

    await page.goto('/#/performance');
    await expect(page.getByText('Backend Director').first()).toBeVisible();
    await expect(page.locator('aside').getByText('Gate ready').first()).toBeVisible();

    await page.goto('/#/settings');
    await expect(page.getByText('supervised_approval_required').first()).toBeVisible();
    await expect(page.getByText('supervised_manual_gate').first()).toBeVisible();
    await expect(page.getByText('fader, gain').first()).toBeVisible();
    await expect(page.getByText('Backend Product State').first()).toBeVisible();
    await expect(page.getByText('Control Center dashboard, Task Queue, Safety Monitor').first()).toBeVisible();
    await expect(page.getByText('Backend Production Mix v1').first()).toBeVisible();
    await expect(page.getByText('Supervised proof evidence').first()).toBeVisible();
    await expect(page.getByText('artifact_available').first()).toBeVisible();
    await expect(page.getByText('age:180s stale_after:86400s stale:false').first()).toBeVisible();
    await expect(page.getByText('read_surface_gaps:10; fixture_paths:10; test_names:25').first()).toBeVisible();
    await expect(page.getByText('gap_surfaces:raw_mixer_state_cache').first()).toBeVisible();
    await expect(page.getByText('next_readback:raw_mixer_state_cache').first()).toBeVisible();
    await expect(page.getByText('source_next_readback:raw_mixer_state_cache').first()).toBeVisible();
    await expect(page.getByText('next_readback_gate:true').first()).toBeVisible();
    await expect(page.getByText('raw_contract:contract_ready_physical_fixture_missing').first()).toBeVisible();
    await expect(page.getByText('raw_contract_live_authority:false').first()).toBeVisible();
    await expect(page.getByText('raw_contract_target_gate:true').first()).toBeVisible();
    await expect(page.getByText('Raw mixer readback contract').first()).toBeVisible();
    await expect(page.getByText('physical_gap_remains:true').first()).toBeVisible();
    await expect(page.getByText('live_authority:false').first()).toBeVisible();
    await expect(page.getByText('Read surface remediation plan').first()).toBeVisible();
    await expect(page.getByText('read_remediation_required').first()).toBeVisible();
    await expect(page.getByText('next_physical_readback:raw_mixer_state_cache').first()).toBeVisible();
    await expect(page.getByText('next_action:remediate_raw_mixer_state_cache').first()).toBeVisible();
    await expect(page.getByText('Autonomy proof evidence: Fader').first()).toBeVisible();
    await expect(page.getByText('Autonomy proof evidence: Gain').first()).toBeVisible();
    await expect(page.getByText('dry_run_only').first()).toBeVisible();
    await page.getByRole('button', { name: 'Module Status' }).first().click();
    await expect(page.getByText('Proof freshness').first()).toBeVisible();
    await expect(page.getByText('Fresh', { exact: true }).first()).toBeVisible();
    await expect(page.getByText('Dry Run Only (2)').first()).toBeVisible();
    await expect(page.getByText('Read-surface proof').first()).toBeVisible();
    await expect(page.getByText('Metadata available').first()).toBeVisible();
    await expect(page.getByText('Read-surface gaps').first()).toBeVisible();
    await expect(page.getByText('10 physical gaps').first()).toBeVisible();
    await expect(page.getByText('Gap surfaces').first()).toBeVisible();
    await expect(page.getByText('raw_mixer_state_cache +9').first()).toBeVisible();
    await expect(page.getByText('Next read proof').first()).toBeVisible();
    await expect(page.getByText('raw_mixer_state_cache', { exact: true }).first()).toBeVisible();
    await expect(page.getByText('Read target gate').first()).toBeVisible();
    await expect(page.getByText('Consistent').first()).toBeVisible();
    await expect(page.getByText('Fixture tests').first()).toBeVisible();
    await expect(page.getByText('25 declared tests').first()).toBeVisible();
    await expect(page.getByText('Contract/gap gate').first()).toBeVisible();
    await expect(page.getByText('Enforced').first()).toBeVisible();
    await expect(page.getByText('Raw mixer contract').first()).toBeVisible();
    await expect(page.getByText('Contract Ready Physical Fixture Missing').first()).toBeVisible();
    await expect(page.getByText('Contract gap').first()).toBeVisible();
    await expect(page.getByText('Physical proof missing').first()).toBeVisible();
    await expect(page.getByText('Contract authority').first()).toBeVisible();
    await expect(page.getByText('No live authority').first()).toBeVisible();
    await expect(page.getByText('Contract target').first()).toBeVisible();

    await page.goto('/#/groups');
    await expect(page.getByText('Backend DTO Drums').first()).toBeVisible();
    await expect(page.getByText('Backend DTO Vocals').first()).toBeVisible();
    await expect(page.getByText('Backend Kick').first()).toBeVisible();
    await expect(page.getByText('groups_buses_snapshot / backend_fixture_groups').first()).toBeVisible();

    await page.goto('/#/devices');
    await expect(page.getByText('Backend WING').first()).toBeVisible();

    await page.goto('/#/system');
    await expect(page.getByText('Backend Core Node').first()).toBeVisible();
    await expect(page.getByText('Backend Mixer Node').first()).toBeVisible();

    await page.goto('/#/routing');
    await expect(page.getByText('Backend Routing Segment').first()).toBeVisible();
    await expect(page.getByText('Backend Tap Point').first()).toBeVisible();

    await page.goto('/#/osc-midi');
    await expect(page.getByText('Backend WING').first()).toBeVisible();
    await expect(page.getByText('supervised_manual_gate').first()).toBeVisible();
    await expect(page.getByText('Direct OSC writes').first()).toBeVisible();

    await page.goto('/#/logs');
    await expect(page.getByText('Dashboard snapshot received.').first()).toBeVisible();
    await expect(page.getByText('Connection topology received.').first()).toBeVisible();

    await page.goto('/#/training');
    await expect(page.getByText('Backend DTO training set').first()).toBeVisible();
    await expect(page.getByText('42').first()).toBeVisible();

    await page.goto('/#/rules');
    await expect(page.getByText('Backend DTO no direct writes').first()).toBeVisible();
    await expect(page.getByText('Backend DTO supervised gate only').first()).toBeVisible();

    expect(sentTypes).toEqual(expect.arrayContaining([
      'get_operator_mode',
      'get_dashboard_snapshot',
      'get_channel_inventory',
      'get_decision_queue',
      'get_decision_traces',
      'get_safety_monitor',
      'get_replay_runs',
      'get_channel_intelligence',
      'get_operator_agent_roster',
      'get_connection_topology',
      'get_groups_buses_snapshot',
      'get_memory_knowledge_snapshot',
      'get_training_sessions_snapshot',
      'get_rules_library_snapshot',
      'get_agent_communications_snapshot',
      'get_processing_pipeline_snapshot',
      'get_main_pipeline_preview',
      'start_replay_run',
      'apply_operator_proposal'
    ]));
    expect(sentTypes.filter(type => type === 'get_replay_runs').length).toBeGreaterThanOrEqual(2);
    expect(sentTypes).not.toEqual(expect.arrayContaining([
      'connect_wing',
      'auto_connect',
      'set_fader',
      'set_gain',
      'set_eq',
      'set_compressor',
      'route_output',
      'route_multiple_outputs',
      'set_channel_input',
      'set_channel_alt_input',
      'load_snap',
      'save_snap',
      'undo_restore_snapshot',
      'set_insert',
      'set_fx_model',
      'set_fx_on',
      'set_fx_mix',
      'set_fx_parameter',
      'reset_phase_delay',
      'start_auto_eq',
      'apply_eq_correction',
      'start_realtime_fader',
      'apply_auto_balance',
      'start_auto_soundcheck',
      'start_auto_compressor',
      'start_auto_compressor_soundcheck',
      'start_auto_compressor_live'
    ]));
  });
});
