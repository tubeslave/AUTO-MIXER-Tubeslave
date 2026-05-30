import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import ts from 'typescript';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const srcRoot = path.resolve(__dirname, '../..');

async function importTypeScript(relativePath) {
  const sourcePath = path.join(srcRoot, relativePath);
  const source = await fs.readFile(sourcePath, 'utf8');
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: {
      jsx: ts.JsxEmit.ReactJSX,
      module: ts.ModuleKind.ES2022,
      target: ts.ScriptTarget.ES2022,
    },
    fileName: sourcePath,
  });
  const encoded = Buffer.from(
    `${outputText}\n//# sourceURL=${pathToFileURL(sourcePath).href}`,
    'utf8',
  ).toString('base64');
  return import(`data:text/javascript;base64,${encoded}`);
}

test('adaptTasks preserves SafeGain ensemble evidence for read-only cockpit display', async () => {
  const { adaptTasks } = await importTypeScript('services/operatorAdapters.ts');

  const [task] = adaptTasks(
    {
      pending_actions: [
        {
          id: 'proposal-safe-gain-1',
          title: 'SafeGain trim review',
          target: 'Lead Vox',
          severity: 'high',
          status: 'pending',
          source: 'SafeGain',
          reason:
            'SafeGain measured input headroom. Ensemble context inactive (ensemble_time_context_stale); time stale; sources 1/2; lag 10/2 samples; operator review only.',
          requested_change: {
            value_type: 'gain',
            channel: 12,
            value: 3,
            safety_evidence: [
              {
                label: 'Ensemble context',
                value: 'ensemble_time_context_stale',
                reason: 'ensemble_time_context_stale',
              },
              {
                label: 'Ensemble time coherence',
                value: 'stale',
                reason: 'ensemble_time_context_stale',
              },
              {
                label: 'Ensemble time sources',
                value: 1,
                target: 2,
              },
              {
                label: 'Ensemble lag samples',
                value: 10,
                target: 2,
              },
              {
                label: 'Ensemble apply policy',
                value: 'operator_review_only',
              },
              {
                label: 'Distortion risk',
                value: 'low',
              },
            ],
          },
          can_apply: true,
        },
      ],
    },
    [],
  );

  assert.equal(task.reason.includes('Ensemble context inactive'), true);
  assert.deepEqual(
    task.safetyEvidence.map(row => row.label),
    [
      'Ensemble context',
      'Ensemble time coherence',
      'Ensemble time sources',
      'Ensemble lag samples',
      'Ensemble apply policy',
      'Distortion risk',
    ],
  );
  assert.equal(task.safetyEvidence[0].value, 'ensemble time context stale');
  assert.equal(task.safetyEvidence[0].tone, 'yellow');
  assert.equal(task.safetyEvidence[2].value, '1/2');
  assert.equal(task.safetyEvidence[3].value, '10/2');
  assert.equal(task.safetyEvidence[4].tone, 'yellow');
  assert.equal(task.safetyEvidence[5].tone, 'green');
});

test('safetyEvidenceForTask keeps ensemble rows visible before generic evidence', async () => {
  const { ensembleEvidenceForTask, safetyEvidenceForTask } = await importTypeScript('mock/automixerData.ts');

  const task = {
    id: 'proposal-safe-gain-1',
    priority: 'High',
    title: 'SafeGain trim review',
    agent: 'SafeGain',
    status: 'Pending',
    time: '10:31',
    safetyEvidence: [
      { label: 'Distortion risk', value: 'low', tone: 'green' },
      { label: 'Ensemble context', value: 'ensemble time context stale', tone: 'yellow' },
      { label: 'Preamp zone', value: 'clean', tone: 'green' },
      { label: 'Ensemble time coherence', value: 'stale', tone: 'yellow' },
      { label: 'Ensemble time sources', value: '1/2', tone: 'yellow' },
      { label: 'Readback', value: 'required', tone: 'yellow' },
      { label: 'Ensemble lag samples', value: '10/2', tone: 'yellow' },
      { label: 'Rollback', value: 'required', tone: 'yellow' },
      { label: 'Ensemble apply policy', value: 'operator review only', tone: 'yellow' },
    ],
  };

  const evidence = safetyEvidenceForTask(task, 'Supervised');

  assert.deepEqual(
    evidence.map(row => row.label),
    [
      'Ensemble context',
      'Ensemble time coherence',
      'Ensemble time sources',
      'Ensemble lag samples',
      'Ensemble apply policy',
      'Distortion risk',
    ],
  );

  assert.deepEqual(
    ensembleEvidenceForTask(task).map(row => row.label),
    [
      'Ensemble context',
      'Ensemble time coherence',
      'Ensemble time sources',
      'Ensemble lag samples',
      'Ensemble apply policy',
    ],
  );
});
