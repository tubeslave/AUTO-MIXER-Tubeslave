export const INSTRUMENT_PRESETS = [
  { id: 'kick', name: 'Kick' },
  { id: 'kick_in', name: 'Kick in' },
  { id: 'kick_out', name: 'Kick out' },
  { id: 'kick_sub', name: 'Kick sub' },
  { id: 'snare', name: 'Snare' },
  { id: 'snare_top', name: 'Snare top' },
  { id: 'snare_bottom', name: 'Snare bottom' },
  { id: 'tom', name: 'Tom' },
  { id: 'tom_hi', name: 'Hi Tom' },
  { id: 'tom_mid', name: 'Mid Tom' },
  { id: 'tom_floor', name: 'Floor Tom' },
  { id: 'hi_hat', name: 'Hi-hat' },
  { id: 'ride', name: 'Ride' },
  { id: 'overhead', name: 'Overhead' },
  { id: 'ambience', name: 'Ambience' },
  { id: 'bass', name: 'Bass' },
  { id: 'electric_guitar', name: 'Electric Guitar' },
  { id: 'acoustic_guitar', name: 'Acoustic Guitar' },
  { id: 'keys', name: 'Keys' },
  { id: 'synth', name: 'Synth' },
  { id: 'playback', name: 'Playback' },
  { id: 'accordion', name: 'Accordion' },
  { id: 'back_vocal', name: 'Back Vocal' },
  { id: 'lead_vocal', name: 'Lead Vocal' },
  { id: 'vocal', name: 'Vocal' },
  { id: 'custom', name: 'Custom' },
];

const DEFAULT_PRESET = 'custom';

const METHOD_PRESET_ALIASES = {
  gain: {
    kick_in: 'kick',
    kick_out: 'kick',
    kick_sub: 'kick',
    snare_top: 'snare',
    tom_floor: 'tom_lo',
    electric_guitar: 'guitar',
    acoustic_guitar: 'guitar',
    back_vocal: 'bgv',
    lead_vocal: 'vocal',
    hi_hat: 'hi_hat',
    ambience_l: 'ambience',
    ambience_r: 'ambience',
  },
  eq: {
    kick_in: 'kick',
    kick_out: 'kick',
    kick_sub: 'kick',
    snare_top: 'snare',
    snare_bottom: 'snare',
    tom_hi: 'tom',
    tom_mid: 'tom',
    tom_floor: 'ftom',
    hi_hat: 'hihat',
    overhead: 'overheads',
    electric_guitar: 'guitar',
    acoustic_guitar: 'acousticguitar',
    back_vocal: 'backvocal',
    lead_vocal: 'leadvocal',
    vocal: 'leadvocal',
  },
  compressor: {
    kick_in: 'kick',
    kick_out: 'kick',
    kick_sub: 'kick',
    snare_top: 'snare',
    tom_hi: 'tom',
    tom_mid: 'tom',
    tom_floor: 'tom',
    hi_hat: 'hihat',
    overhead: 'overheads',
    electric_guitar: 'electricGuitar',
    acoustic_guitar: 'acousticGuitar',
    back_vocal: 'backVocal',
    lead_vocal: 'leadVocal',
    vocal: 'leadVocal',
  },
  generic: {
    snare_top: 'snare',
    kick_in: 'kick',
    kick_out: 'kick',
    kick_sub: 'kick',
    tom_floor: 'tom',
    electric_guitar: 'electricGuitar',
    acoustic_guitar: 'acousticGuitar',
    back_vocal: 'backVocal',
    lead_vocal: 'leadVocal',
    hi_hat: 'hihat',
  },
};

export function mapPresetForMethod(presetId, method = 'generic') {
  const raw = String(presetId || DEFAULT_PRESET);
  const aliases = METHOD_PRESET_ALIASES[method] || METHOD_PRESET_ALIASES.generic;
  return aliases[raw] || raw;
}

export function detectInstrumentPreset(channelName) {
  if (!channelName || !String(channelName).trim()) return DEFAULT_PRESET;
  const n = String(channelName).toLowerCase().trim();

  if (/\b(kick\s*in|bd\s*in)\b/i.test(n)) return 'kick_in';
  if (/\b(kick\s*out|bd\s*out)\b/i.test(n)) return 'kick_out';
  if (/\b(kick\s*sub|sub[\s-]?kick)\b/i.test(n)) return 'kick_sub';
  if (/\b(kick|bd|bass\s*drum|бочка|кик)\b/i.test(n)) return 'kick';
  if (/\b(snare\s*top|sn\s*top)\b/i.test(n)) return 'snare_top';
  if (/\b(snare\s*bottom|snare\s*bot|sn\s*bottom)\b/i.test(n)) return 'snare_bottom';
  if (/\b(snare|sd|sn|малый|снэйр)\b/i.test(n)) return 'snare';
  if (/\b(hi\s*tom|high\s*tom)\b/i.test(n)) return 'tom_hi';
  if (/\b(mid\s*tom|middle\s*tom)\b/i.test(n)) return 'tom_mid';
  if (/\b(floor\s*tom|low\s*tom|ftom|флор)\b/i.test(n)) return 'tom_floor';
  if (/\b(tom|том)\b/i.test(n)) return 'tom';
  if (/\b(hi[\s-]?hat|hh|хай[\s-]?хэт)\b/i.test(n)) return 'hi_hat';
  if (/\b(ride|райд)\b/i.test(n)) return 'ride';
  if (/\b(ohl|ohr|over[\s-]?head|overhead)\b/i.test(n)) return 'overhead';
  if (/\b(ambience|ambient|room\s*mic|зал|амбьенс)\b/i.test(n)) return 'ambience';
  if (/\b(bass|бас|sub)(?![\s-]?(drum|бочка))/i.test(n)) return 'bass';
  if (/\b(acoustic|акустик|agtr)\b/i.test(n)) return 'acoustic_guitar';
  if (/\b(electric|электро|egtr|gtr|гитар|guitar)\b/i.test(n)) return 'electric_guitar';
  if (/\b(keys|keyboard|piano|клавиш)\b/i.test(n)) return 'keys';
  if (/\b(synth|синт)\b/i.test(n)) return 'synth';
  if (/\b(playback|pb|track|backing|минус)\b/i.test(n)) return 'playback';
  if (/\b(accordion|accord|bayan|баян|аккордеон)\b/i.test(n)) return 'accordion';
  if (/\b(back[\s-]?vox|bvox|бэк[\s-]?вок|choir|хор|bgv)\b/i.test(n)) return 'back_vocal';
  if (/\b(lead\s*vox|lead\s*vocal|лид[\s-]?вок|main\s*vox|vox\s*1)\b/i.test(n)) return 'lead_vocal';
  if (/\b(vox|vocal|вокал|голос|voice|mic)\b/i.test(n)) return 'vocal';
  return DEFAULT_PRESET;
}
