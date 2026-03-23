"""
Channel name recognition module.
Detects instrument/source type from channel names.
"""

import re
import logging
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

# Маппинг ключевых слов на preset ID
# Формат: (паттерн regex, preset_id)
# Паттерны проверяются в порядке приоритета (более специфичные первые)

KEYWORD_PATTERNS = [
    # Drums - specific
    (r'\b(kick|bd|bass\s*drum|бас[\s-]?бочка|бочка|кик)\b', 'kick'),
    (r'\b(snare|sd|sn|малый|снэйр|снейр)\b', 'snare'),
    (r'\b(hi[\s-]?hat|hh|хай[\s-]?хэт|хэт)\b', 'hihat'),
    (r'\b(ride|райд)\b', 'ride'),
    (r'\b(tom|том|floor|флор)\b', 'tom'),
    (r'\b(crash|splash|china|крэш|сплэш|чайна|cymbal|тарелк)\b', 'cymbals'),
    (r'\b(oh|ohl|ohr|over[\s-]?head|overhead|оверхэд)\b', 'overheads'),
    (r'\b(room|рум)\b', 'room'),  # Room mic
    
    # Bass
    (r'\b(bass|бас|sub|саб)\b(?![\s-]?(drum|бочка))', 'bass'),
    
    # Guitars
    (r'\b(acoustic|акустик|акуст[\s-]?гитар|акуст|agtr)\b', 'acousticGuitar'),
    (r'\b(electric|электро|egtr|e[\s-]?gtr|gtr|гитар|guitar)\b', 'electricGuitar'),
    
    # Keys & Instruments
    (r'\b(accordion|accord|bayan|баян|аккордеон|гармонь|гармошка)\b', 'accordion'),
    (r'\b(synth|keys|keyboard|piano|клавиш|синт|пиано|орган|organ|rhodes|wurli)\b', 'synth'),
    (r'\b(playback|pb|track|backing|минус|фонограмма|плейбэк|трек)\b', 'playback'),
    
    # Vocals - check for names first (common Russian/English names)
    (r'\b(lead\s*vox|lead\s*vocal|лид[\s-]?вок|main\s*vox|solo\s*vox|соло[\s-]?вок|vox\s*1|вок\s*1)\b', 'leadVocal'),
    (r'\b(back[\s-]?vox|backing[\s-]?vox|bvox|бэк[\s-]?вок|choir|хор|bgv|vox\s*[2-9]|вок\s*[2-9])\b', 'backVocal'),
    # Common names that should be recognized as vocals
    (r'\b(katya|катя|sergey|сергей|slava|слава|dima|дима|masha|маша|sasha|саша|pasha|паша|vova|вова|andrey|андрей|alex|алекс|misha|миша|natasha|наташа|olga|ольга|tanya|таня|vlad|влад|ivan|иван|max|макс|nikita|никита|dasha|даша|anya|аня|lena|лена|maria|мария|anna|анна|elena|елена)\b', 'leadVocal'),
    (r'\b(vox|vocal|вокал|голос|voice|mic|микрофон)\b', 'leadVocal'),  # Generic vocal defaults to lead
    # Bus / DCA
    (r'\b(drums?\s*bus|drum\s*grp|группа\s*барабан|барабаны\s*бус)\b', 'drums_bus'),
    (r'\b(vocal\s*bus|vox\s*bus|вокал\s*бус|голос\s*бус)\b', 'vocal_bus'),
    (r'\b(instr\s*bus|instrument\s*bus|инструмент\s*бус|бус\s*инстр)\b', 'instrument_bus'),
]

def recognize_instrument(channel_name: str) -> Optional[str]:
    """
    Recognize instrument type from channel name.
    
    Args:
        channel_name: Channel name from mixer
        
    Returns:
        preset_id if recognized, None otherwise
    """
    if not channel_name:
        return None
    
    name_lower = channel_name.lower().strip()
    
    for pattern, preset_id in KEYWORD_PATTERNS:
        if re.search(pattern, name_lower, re.IGNORECASE):
            logger.debug(f"Channel '{channel_name}' matched pattern '{pattern}' -> {preset_id}")
            return preset_id
    
    return None


def recognize_instrument_spectral_fallback(
    channel_name: str,
    centroid_hz: float = 0.0,
    energy_bands: Optional[Dict[str, float]] = None,
) -> Optional[str]:
    """
    Fallback: если по имени не распознан, классификация по спектральному отпечатку
    (centroid, energy bands) за первые секунды саундчека. Заглушка с простой эвристикой.
    """
    if energy_bands is None:
        energy_bands = {}
    low = energy_bands.get("low_100_300", 0.0)
    mid = energy_bands.get("mid_1k_4k", 0.0)
    high = energy_bands.get("high_4k_10k", 0.0)
    if centroid_hz <= 0 and not energy_bands:
        return None
    if centroid_hz > 0:
        if centroid_hz < 200 and low > 0.5:
            return "kick"
        if 200 <= centroid_hz < 800 and low > 0.3:
            return "bass"
        if 2000 <= centroid_hz < 6000 and mid > 0.4:
            return "leadVocal"
        if centroid_hz >= 6000 and high > 0.4:
            return "hihat"
    return None


def scan_and_recognize(channel_names: Dict[int, str]) -> Dict[int, Dict]:
    """
    Scan channel names and recognize instruments.
    
    Args:
        channel_names: Dict mapping channel number to name
        
    Returns:
        Dict with recognition results:
        {
            channel_num: {
                'name': 'Kick',
                'preset': 'kick',
                'recognized': True
            }
        }
    """
    results = {}
    
    for channel_num, name in channel_names.items():
        preset = recognize_instrument(name)
        results[channel_num] = {
            'name': name,
            'preset': preset,
            'recognized': preset is not None
        }
        
        if preset:
            logger.info(f"Channel {channel_num} '{name}' -> {preset}")
        else:
            logger.debug(f"Channel {channel_num} '{name}' -> not recognized")
    
    recognized_count = sum(1 for r in results.values() if r['recognized'])
    logger.info(f"Recognition complete: {recognized_count}/{len(results)} channels recognized")
    
    return results

# Список всех доступных пресетов для справки
AVAILABLE_PRESETS = {
    'kick': 'Kick',
    'snare': 'Snare', 
    'tom': 'Tom',
    'hihat': 'Hi-Hat',
    'ride': 'Ride',
    'cymbals': 'Cymbals',
    'overheads': 'Overheads',
    'room': 'Room',
    'bass': 'Bass',
    'electricGuitar': 'Electric Guitar',
    'acousticGuitar': 'Acoustic Guitar',
    'accordion': 'Accordion',
    'synth': 'Synth / Keys',
    'playback': 'Playback',
    'leadVocal': 'Lead Vocal',
    'backVocal': 'Back Vocal',
    'drums_bus': 'Drums Bus',
    'vocal_bus': 'Vocal Bus',
    'instrument_bus': 'Instrument Bus',
    'custom': 'Custom'
}
