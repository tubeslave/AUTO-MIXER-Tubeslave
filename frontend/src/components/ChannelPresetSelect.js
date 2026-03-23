import React from 'react';
import { INSTRUMENT_PRESETS } from '../constants/instrumentPresets';

/**
 * Единый выпадающий список инструментального пресета для канала.
 */
function ChannelPresetSelect({ value, onChange, disabled = false, className = '' }) {
  return (
    <select
      className={`channel-preset-select ${className}`.trim()}
      value={value || 'custom'}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      onClick={(e) => e.stopPropagation()}
    >
      {INSTRUMENT_PRESETS.map((p) => (
        <option key={p.id} value={p.id}>
          {p.name}
        </option>
      ))}
    </select>
  );
}

export default ChannelPresetSelect;
