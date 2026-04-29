function topEntries(counts = {}, limit = 3) {
  return Object.entries(counts)
    .filter(([, value]) => Number.isFinite(value) && value > 0)
    .sort((left, right) => {
      if (right[1] !== left[1]) return right[1] - left[1];
      return left[0].localeCompare(right[0]);
    })
    .slice(0, limit);
}

function formatCountEntries(entries) {
  return entries.map(([name, count]) => `${name} (${count})`).join(', ');
}

function sectionFromEntries(title, counts = {}, limit = 3) {
  const entries = topEntries(counts, limit);
  if (!entries.length) {
    return null;
  }
  return {
    title,
    items: entries.map(([name, count]) => `${name} (${count})`),
  };
}

export function buildAutofohSessionSummary(status = {}) {
  const report = status.autofoh_session_report || null;
  const summaryText = status.autofoh_session_report_summary || '';
  const reportPath = status.autofoh_report_path || '';

  if (!report && !summaryText) {
    return null;
  }

  const chips = [];
  const detailLines = [];
  const sections = [];

  if (report) {
    chips.push({ label: 'Sent', value: report.action_sent_count || 0 });
    chips.push({ label: 'Blocked', value: report.action_blocked_count || 0 });
    chips.push({ label: 'Guard Blocks', value: report.guard_block_count || 0 });

    if ((report.evaluation_count || 0) > 0) {
      chips.push({ label: 'Evaluations', value: report.evaluation_count || 0 });
    }
    if ((report.rollback_count || 0) > 0) {
      chips.push({ label: 'Rollbacks', value: report.rollback_count || 0 });
    }

    const guardedChannels = Array.isArray(report.channels_with_guard_blocks)
      ? report.channels_with_guard_blocks
      : [];
    if (guardedChannels.length) {
      const visibleChannels = guardedChannels.slice(0, 6).map(channel => `Ch ${channel}`);
      const overflow = guardedChannels.length > 6 ? ` +${guardedChannels.length - 6}` : '';
      detailLines.push(`Guarded channels: ${visibleChannels.join(', ')}${overflow}`);
      sections.push({
        title: 'Guarded Channels',
        items: guardedChannels.map(channel => `Ch ${channel}`),
      });
    }

    const topReasons = topEntries(report.guard_blocks_by_reason, 2);
    if (topReasons.length) {
      detailLines.push(`Top guard reasons: ${formatCountEntries(topReasons)}`);
    }
    const guardReasonSection = sectionFromEntries('Top Guard Reasons', report.guard_blocks_by_reason, 4);
    if (guardReasonSection) sections.push(guardReasonSection);

    const topActionTypes = topEntries(report.guard_blocks_by_action_type, 2);
    if (topActionTypes.length) {
      detailLines.push(`Blocked action types: ${formatCountEntries(topActionTypes)}`);
    }
    const blockedActionSection = sectionFromEntries('Blocked Action Types', report.guard_blocks_by_action_type, 4);
    if (blockedActionSection) sections.push(blockedActionSection);

    const topPhases = topEntries(report.guard_blocks_by_phase, 1);
    if (topPhases.length) {
      const [phaseName, phaseCount] = topPhases[0];
      detailLines.push(`Most restrictive phase: ${phaseName} (${phaseCount})`);
    }
    const phaseSection = sectionFromEntries('Runtime Phases', report.guard_blocks_by_phase, 3);
    if (phaseSection) sections.push(phaseSection);

    const runtimeStateSection = sectionFromEntries('Runtime States', report.guard_blocks_by_runtime_state, 3);
    if (runtimeStateSection) sections.push(runtimeStateSection);
  }

  if (reportPath) {
    detailLines.push(`Report: ${reportPath}`);
    sections.push({
      title: 'Session Report',
      items: [reportPath],
    });
  }

  return {
    title: 'AutoFOH Session Summary',
    summaryText: summaryText || 'Session report available.',
    chips,
    detailLines,
    sections,
    hasExpandableDetails: sections.length > 0,
  };
}
