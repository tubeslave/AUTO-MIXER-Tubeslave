"""
Voice Control Module for Auto Mixer
Uses Faster-Whisper for Speech-to-Text recognition
With fuzzy matching for better command recognition
"""
import asyncio
import logging
import pyaudio
import numpy as np
import re
import time
from typing import Optional, Callable, Dict, Any, List, Tuple
from faster_whisper import WhisperModel
import threading
import queue
from rapidfuzz import fuzz, process

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceControl:
    """
    Voice control handler using Faster-Whisper STT
    
    Supports real-time voice command recognition and mapping to mixer controls
    """
    
    def __init__(self, 
                 model_size: str = "small",
                 device: str = "cpu",
                 compute_type: str = "int8",
                 language: Optional[str] = "ru",
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 input_device_index: Optional[int] = None,
                 input_channel: int = 0):
        """
        Initialize voice control
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (cpu, cuda)
            compute_type: Compute type (int8, int8_float16, float16, float32)
            language: Language code (ru, en, etc.) or None for auto-detection
            sample_rate: Audio sample rate (Hz)
            chunk_size: Audio chunk size for recording
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.input_device_index = input_device_index
        self.input_channel = input_channel
        
        self.model: Optional[WhisperModel] = None
        self.audio = None
        self.stream = None
        
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.command_callback: Optional[Callable[[str], None]] = None
        
        # Command mapping: voice command -> mixer action
        self.command_map = self._init_command_map()
        
        logger.info(f"VoiceControl initialized (model: {model_size}, language: {language})")
    
    def _init_command_map(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize command mapping dictionary
        
        Maps voice commands to mixer actions
        """
        return {
            # Channel fader commands
            "канал": {
                "pattern": r"канал\s+(\d+)",
                "action": "set_fader"
            },
            "channel": {
                "pattern": r"channel\s+(\d+)",
                "action": "set_fader"
            },
            # Gain commands
            "гейн": {
                "pattern": r"гейн\s+(\d+)",
                "action": "set_gain"
            },
            "gain": {
                "pattern": r"gain\s+(\d+)",
                "action": "set_gain"
            },
            # Snapshot commands
            "снапшот": {
                "pattern": r"снапшот\s+(.+)",
                "action": "load_snap"
            },
            "snapshot": {
                "pattern": r"snapshot\s+(.+)",
                "action": "load_snap"
            },
            "загрузить": {
                "pattern": r"загрузить\s+(.+)",
                "action": "load_snap"
            },
            "load": {
                "pattern": r"load\s+(.+)",
                "action": "load_snap"
            },
            # Mute commands
            "мут": {
                "pattern": r"мут\s+(\d+)",
                "action": "mute_channel"
            },
            "mute": {
                "pattern": r"mute\s+(\d+)",
                "action": "mute_channel"
            },
            # Volume up/down - more flexible patterns
            "громче": {
                "pattern": r"громче",
                "action": "volume_up"
            },
            "тише": {
                "pattern": r"тише",
                "action": "volume_down"
            },
            "louder": {
                "pattern": r"louder",
                "action": "volume_up"
            },
            "quieter": {
                "pattern": r"quieter",
                "action": "volume_down"
            },
            # More flexible channel patterns
            "первый": {
                "pattern": r"первый\s+канал",
                "action": "set_fader"
            },
            "второй": {
                "pattern": r"второй\s+канал",
                "action": "set_fader"
            },
            "третий": {
                "pattern": r"третий\s+канал",
                "action": "set_fader"
            },
            "фейдер": {
                "pattern": r"фейдер\s+(?:канал\s+)?(?:первый|второй|третий|четвертый|пятый|\d+)",
                "action": "set_fader"
            }
        }
    
    def load_model(self):
        """Load Whisper model"""
        try:
            logger.info("=" * 60)
            logger.info(f"LOADING WHISPER MODEL: {self.model_size} on {self.device}")
            logger.info("This may take a while on first load...")
            logger.info("=" * 60)
            
            # Load model - this might block for a moment
            import time
            start_time = time.time()
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info(f"✅ WHISPER MODEL LOADED SUCCESSFULLY in {elapsed:.2f} seconds")
            logger.info("=" * 60)
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ ERROR LOADING WHISPER MODEL: {e}")
            logger.error("=" * 60)
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def start_listening(self, callback: Callable[[str], None]):
        """
        Start listening for voice commands
        
        Args:
            callback: Function to call when a command is recognized
        """
        logger.info("=" * 60)
        logger.info("START_LISTENING CALLED")
        logger.info("=" * 60)
        
        if self.is_listening:
            logger.warning("Already listening")
            return
        
        # Don't load model here if it's already loaded
        # Model should be loaded before calling start_listening
        if not self.model:
            logger.warning("Model not loaded! Should be loaded before start_listening")
            try:
                logger.info("Loading Whisper model (this may take a moment on first run)...")
                self.load_model()
                logger.info("✅ Model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}", exc_info=True)
                raise
        
        logger.info("Setting callback and is_listening flag...")
        self.command_callback = callback
        self.is_listening = True
        logger.info("✅ is_listening set to True")
        
        # Initialize PyAudio
        try:
            self.audio = pyaudio.PyAudio()
            
            # Determine number of channels for the device
            input_channels = 1
            if self.input_device_index is not None:
                try:
                    device_info = self.audio.get_device_info_by_index(self.input_device_index)
                    max_channels = int(device_info.get('maxInputChannels', 1))
                    # Use mono for voice recognition, but record from multi-channel if needed
                    input_channels = 1  # Always mono for Whisper
                    logger.info(f"Opening audio device {self.input_device_index} (max channels: {max_channels}, using channel {self.input_channel})")
                except Exception as e:
                    logger.warning(f"Could not get device info: {e}, using mono")
            
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=input_channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.stream.start_stream()
            logger.info(f"Audio stream started on device {self.input_device_index}, channel {self.input_channel}")
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._process_audio, daemon=True)
            self.processing_thread.start()
            
            logger.info("Voice control started listening")
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            self.is_listening = False
            if self.audio:
                self.audio.terminate()
            raise
    
    def stop_listening(self):
        """Stop listening for voice commands"""
        if not self.is_listening:
            return
        
        self.is_listening = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        logger.info("Voice control stopped listening")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if self.is_listening:
            # For mono input (channels=1), use data as-is
            # For multi-channel, we'll handle it in the stream setup
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def _process_audio(self):
        """Process audio chunks and recognize speech"""
        audio_buffer = []
        buffer_duration = 4.0  # Buffer duration in seconds
        buffer_size = int(self.sample_rate * buffer_duration * 2)  # *2 for int16
        last_speech_time = time.time()
        silence_threshold = 2.0  # seconds of silence before processing
        
        while self.is_listening:
            try:
                # Collect audio chunks
                chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer.append(chunk)
                current_time = time.time()
                
                # Check if we have enough audio or enough silence
                buffer_bytes = len(b''.join(audio_buffer))
                time_since_last_speech = current_time - last_speech_time
                
                # Process if buffer is full OR if we have silence after speech
                should_process = (buffer_bytes >= buffer_size) or (
                    buffer_bytes > int(self.sample_rate * 1.0) and  # At least 1 second of audio
                    time_since_last_speech > silence_threshold
                )
                
                if should_process and buffer_bytes > 0:
                    audio_data = np.frombuffer(b''.join(audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Reset buffer
                    audio_buffer = []
                    last_speech_time = current_time
                    
                    # Transcribe with improved parameters for Russian
                    segments, info = self.model.transcribe(
                        audio_data,
                        language=self.language,
                        beam_size=5,
                        best_of=5,
                        temperature=0.0,  # More deterministic
                        vad_filter=True,
                        vad_parameters=dict(
                            min_silence_duration_ms=600,  # Balance between cutting and waiting
                            threshold=0.5,  # Higher threshold = less noise
                            speech_pad_ms=300  # Pad speech segments
                        ),
                        no_speech_threshold=0.6,  # Reject segments with high no-speech probability
                        log_prob_threshold=-1.0,  # Reject low confidence segments
                        compression_ratio_threshold=2.4,  # Reject repetitive/hallucinatory text
                    )
                    
                    # Process segments - filter by confidence
                    text_parts = []
                    for segment in segments:
                        text = segment.text.strip().lower()
                        # Filter out low-confidence and short segments
                        if text and len(text) > 1:
                            # Skip segments that are likely noise
                            if segment.no_speech_prob < 0.5:  # Only accept if likely speech
                                text_parts.append(text)
                                last_speech_time = current_time
                    
                    if text_parts:
                        full_text = " ".join(text_parts)
                        logger.info(f"Recognized (raw): {full_text}")
                        
                        # Post-process text to fix common recognition errors
                        corrected_text = self._correct_recognition_errors(full_text)
                        if corrected_text != full_text:
                            logger.info(f"Recognized (corrected): {corrected_text}")
                        
                        # Only process if we have meaningful text (more than 2 chars)
                        if len(corrected_text) > 2:
                            # Parse command
                            command = self._parse_command(corrected_text)
                            if command and self.command_callback:
                                try:
                                    self.command_callback(command)
                                except Exception as e:
                                    logger.error(f"Error in command callback: {e}", exc_info=True)
                
            except queue.Empty:
                # Check if we should process accumulated buffer after silence
                if len(audio_buffer) > 0:
                    current_time = time.time()
                    time_since_last_speech = current_time - last_speech_time
                    buffer_bytes = len(b''.join(audio_buffer))
                    
                    if buffer_bytes > int(self.sample_rate * 1.0) and time_since_last_speech > silence_threshold:
                        # Process accumulated buffer
                        audio_data = np.frombuffer(b''.join(audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0
                        audio_buffer = []
                        
                        try:
                            segments, info = self.model.transcribe(
                                audio_data,
                                language=self.language,
                                beam_size=5,
                                best_of=5,
                                temperature=0.0,
                                vad_filter=True,
                                vad_parameters=dict(
                                    min_silence_duration_ms=600,
                                    threshold=0.5,
                                    speech_pad_ms=300
                                ),
                                no_speech_threshold=0.6,
                                log_prob_threshold=-1.0,
                                compression_ratio_threshold=2.4,
                            )
                            
                            text_parts = []
                            for segment in segments:
                                text = segment.text.strip().lower()
                                if text and len(text) > 1 and segment.no_speech_prob < 0.5:
                                    text_parts.append(text)
                            
                            if text_parts:
                                full_text = " ".join(text_parts)
                                logger.info(f"Recognized (raw, after silence): {full_text}")
                                corrected_text = self._correct_recognition_errors(full_text)
                                if corrected_text != full_text:
                                    logger.info(f"Recognized (corrected, after silence): {corrected_text}")
                                
                                if len(corrected_text) > 2:
                                    command = self._parse_command(corrected_text)
                                    if command and self.command_callback:
                                        try:
                                            self.command_callback(command)
                                        except Exception as e:
                                            logger.error(f"Error in command callback: {e}", exc_info=True)
                        except Exception as e:
                            logger.error(f"Error processing audio after silence: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing audio: {e}", exc_info=True)
    
    def _correct_recognition_errors(self, text: str) -> str:
        """
        Correct common recognition errors for Russian commands using fuzzy matching
        """
        # Known command keywords for fuzzy matching
        KEYWORDS = {
            "канал": ["канал", "канала", "каналу", "канале", "канальн"],
            "фейдер": ["фейдер", "фейдера", "федер", "фэйдер"],
            "тише": ["тише", "тишина", "тишин", "тишь", "толтише", "титтише", "тихо", "тихе"],
            "громче": ["громче", "громко", "громкость", "громкост", "громк", "гром"],
            "убрать": ["убрать", "убрат", "убери", "убавь", "убавить", "убав"],
            "прибавь": ["прибавь", "прибавить", "прибав", "добавь", "добавить", "добав"],
            "минус": ["минус", "минусе", "мину|"],
            "плюс": ["плюс", "плюсе"],
            "децибела": ["децибела", "децибел", "дб", "дицы", "дится", "децибелла", "децибел"],
            "первый": ["первый", "первая", "первое", "первому", "первой"],
            "второй": ["второй", "вторая", "второе", "второму", "второй"],
            "третий": ["третий", "третья", "третье", "третьей"],
            "четвертый": ["четвертый", "четвёртый", "четвертая", "четвёртая"],
            "пятый": ["пятый", "пятая", "пятое"],
            "гейн": ["гейн", "гейна", "ген", "геин", "gain"],
            "мут": ["мут", "мьют", "мют", "заглуши", "заглушить"],
            "загрузить": ["загрузить", "загрузи", "загружай", "загруз"],
            "снапшот": ["снапшот", "снэпшот", "снап", "snapshot"],
            # Numbers - keep as-is to avoid fuzzy matching issues
            "двадцать": ["двадцать"],
            "тридцать": ["тридцать"],
            "три": ["три"],
            "пять": ["пять"],
            "десять": ["десять"],
        }
        
        corrected = text.lower()
        
        # Static corrections first
        corrections = {
            # Dash/hyphen as minus (important for "канал – 20")
            r"[–—−]": "минус",  # Various dash types → минус
            r"\s+-\s+": " минус ",  # " - " → " минус "
            # Volume keywords
            r"\bтолтише\b": "тише",
            r"\bтишина\b": "тише",
            r"\bтишин\b": "тише",
            r"\bтиши\b": "тише",
            r"\bтихо\b": "тише",
            r"\bгромко\b": "громче",
            r"\bгром\b": "громче",
            r"\bубрат\b": "убрать",
            r"\bубери\b": "убрать",
            r"\bубав\w*\b": "убрать",
            r"\bприбав\w*\b": "прибавь",
            r"\bдобав\w*\b": "прибавь",
            # dB corrections
            r"дицы\s*белого": "децибела",
            r"дицы\s*белая": "децибела",
            r"дицы\s*белое": "децибела",
            r"дицы\s*бел\w*": "децибела",
            r"дится\s*бел\w*": "децибела",
            r"\bдится\b": "децибела",
            r"\bдицы\b": "децибела",
            r"3d\b": "три дб",
            # Number corrections
            r"\bтрениться\b": "три",
            r"\bтренится\b": "три",
            # Channel corrections  
            r"\bка\.+\b": "канал",  # "ка..." -> "канал"
            r"\bканала\b": "канал",
            r"\bпервой\b": "первый",
            # Noise removal - common false recognitions
            r"\bбелое\b": "",
            r"\bбелая\b": "",
            r"\bбелый\b": "",
            r"\bсыбила\b": "",
            r"\bпиццы\b": "",
            r"\bсделать\s+": "",
            r"\bделать\s+": "",
            r"\bстал\s+": "",
            r"\bсубтитры\b": "",
            r"\bпонара\b": "",
            r"\bпрямо\b": "",
            r"\bвлево\b": "",
            r"\bпусть\b": "",
            r"\bстарый\b": "",
            r"\bпиши\b": "",
            r"\bно\b$": "",  # "но" only at end
            r"\.$": "",
            r"!$": "",
        }
        
        for pattern, replacement in corrections.items():
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        
        # Fuzzy match individual words
        words = corrected.split()
        corrected_words = []
        
        for word in words:
            if len(word) < 3:
                corrected_words.append(word)
                continue
            
            # Check if word matches any known keyword
            best_match = None
            best_score = 0
            
            for canonical, variants in KEYWORDS.items():
                for variant in variants:
                    score = fuzz.ratio(word, variant)
                    if score > best_score and score >= 75:  # 75% similarity threshold
                        best_score = score
                        best_match = canonical
            
            if best_match and best_score >= 75:
                corrected_words.append(best_match)
            else:
                corrected_words.append(word)
        
        corrected = ' '.join(corrected_words)
        
        # Normalize spacing
        corrected = re.sub(r'\s+', ' ', corrected).strip()
        
        return corrected
    
    def _word_to_number(self, word: str) -> Optional[int]:
        """Convert Russian ordinal words to numbers"""
        ordinal_map = {
            "первый": 1, "первая": 1, "первое": 1,
            "второй": 2, "вторая": 2, "второе": 2,
            "третий": 3, "третья": 3, "третье": 3,
            "четвертый": 4, "четвертая": 4, "четвертое": 4,
            "пятый": 5, "пятая": 5, "пятое": 5,
            "шестой": 6, "шестая": 6, "шестое": 6,
            "седьмой": 7, "седьмая": 7, "седьмое": 7,
            "восьмой": 8, "восьмая": 8, "восьмое": 8,
            "девятый": 9, "девятая": 9, "девятое": 9,
            "десятый": 10, "десятая": 10, "десятое": 10,
            "один": 1, "два": 2, "три": 3, "четыре": 4, "пять": 5,
            "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10
        }
        return ordinal_map.get(word.lower())
    
    def _extract_channel_number(self, text: str) -> Optional[int]:
        """Extract channel number from text, handling ordinals and digits"""
        text_lower = text.lower()
        
        # First, try to find ordinal words (they are more specific for channel)
        # Check for patterns like "первый канал" or "канал первый"
        ordinal_words = ["первый", "первая", "первое", "второй", "вторая", "второе",
                         "третий", "третья", "третье", "четвертый", "четвертая", "четвертое",
                         "пятый", "пятая", "пятое", "шестой", "шестая", "шестое",
                         "седьмой", "седьмая", "седьмое", "восьмой", "восьмая", "восьмое",
                         "девятый", "девятая", "девятое", "десятый", "десятая", "десятое"]
        
        for word in ordinal_words:
            if word in text_lower:
                num = self._word_to_number(word)
                if num:
                    return num
        
        # Then try pattern "канал X" where X is a number
        channel_pattern = re.search(r'канал\s+(\d+)', text_lower)
        if channel_pattern:
            return int(channel_pattern.group(1))
        
        # Then try pattern "X канал" where X is a number
        channel_pattern2 = re.search(r'(\d+)\s+канал', text_lower)
        if channel_pattern2:
            return int(channel_pattern2.group(1))
        
        # Try cardinal numbers before digits (to avoid "минус 3" being interpreted as channel)
        cardinal_words = ["один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять"]
        # Only use cardinal if it's near "канал"
        for word in cardinal_words:
            pattern = rf'{word}\s+канал|канал\s+{word}'
            if re.search(pattern, text_lower):
                num = self._word_to_number(word)
                if num:
                    return num
        
        # Last resort: try direct number match, but only if not preceded by "минус" or "плюс"
        # This avoids interpreting "минус 3" as channel 3
        number_match = re.search(r'(?<!минус\s)(?<!плюс\s)(?<!на\s)(\d+)', text_lower)
        if number_match:
            return int(number_match.group(1))
        
        return None
    
    def _extract_db_amount(self, text: str) -> Optional[int]:
        """Extract decibel amount from text"""
        # Order matters - check longer words first to avoid partial matches
        number_words = [
            ("тридцать", 30), ("двадцать", 20), ("десять", 10),
            ("девять", 9), ("восемь", 8), ("семь", 7), ("шесть", 6),
            ("пять", 5), ("четыре", 4), ("три", 3), ("два", 2), ("один", 1),
        ]
        
        # Look for number after минус/плюс/на
        match = re.search(r'(?:минус|плюс|на)\s+(\d+)', text)
        if match:
            return int(match.group(1))
        
        # Look for number word after минус/плюс/на (check longer words first)
        for word, value in number_words:
            if re.search(rf'(?:минус|плюс|на)\s+{word}', text):
                return value
        
        # Look for number before "дб" or "децибела"
        match = re.search(r'(\d+)\s*(?:дб|децибела|децибел)', text)
        if match:
            return int(match.group(1))
        
        # Look for number word before "дб" or "децибела" (e.g. "три дб")
        for word, value in number_words:
            if re.search(rf'{word}\s*(?:дб|децибела|децибел)', text):
                return value
        
        return None
    
    def _parse_command(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse recognized text into command
        
        Args:
            text: Recognized text
            
        Returns:
            Command dictionary or None if no command found
        """
        text = text.lower().strip()
        
        # List of garbage words that should be ignored entirely
        GARBAGE_WORDS = {
            "пиши", "старый", "пусть", "прямо", "влево", "субтитры", 
            "белое", "белая", "белый", "понара", "сыбила", "пиццы",
            "и", "а", "но", "то", "ну", "да", "нет", "ок", "так",
            "это", "вот", "что", "как", "где", "когда", "почему",
            "можно", "нужно", "надо", "хорошо", "ладно", "понял",
        }
        
        # Skip too short or garbage text
        if len(text) < 3 or text.count(' ') > 10:
            return None
        
        # Check if entire text is garbage
        words = text.split()
        meaningful_words = [w for w in words if w not in GARBAGE_WORDS and len(w) > 1]
        if not meaningful_words:
            return None
        
        logger.info(f"Parsing command: '{text}'")
        
        # PRIORITY 1: Check for volume adjustment commands
        # Keywords for volume DOWN
        down_keywords = ["тише", "минус", "убрать", "убавь", "убавить", "убери", "меньше", "quieter", "down", "-"]
        # Keywords for volume UP  
        up_keywords = ["громче", "плюс", "прибавь", "прибавить", "добавь", "добавить", "больше", "louder", "up", "+"]
        
        has_down = any(word in text for word in down_keywords)
        has_up = any(word in text for word in up_keywords)
        
        if has_down or has_up:
            channel = self._extract_channel_number(text)
            db_amount = self._extract_db_amount(text)
            
            # Also try to find number words for dB (check longer words first)
            if db_amount is None:
                for word, value in [("двадцать", 20), ("тридцать", 30), ("десять", 10),
                                   ("девять", 9), ("восемь", 8), ("семь", 7), ("шесть", 6),
                                   ("пять", 5), ("четыре", 4), ("три", 3), ("два", 2), ("один", 1)]:
                    if word in text:
                        db_amount = value
                        break
            
            if db_amount is None:
                db_amount = 3  # Default 3 dB
            
            amount = db_amount / 100.0
            
            # Determine direction - DOWN takes priority if both present (e.g. "убрать громкость")
            is_down = has_down
            
            cmd_type = "volume_down" if is_down else "volume_up"
            logger.info(f"✅ Parsed {cmd_type} command: channel={channel}, amount={db_amount} dB")
            
            return {
                "type": cmd_type,
                "channel": channel,
                "amount": amount,
                "db": db_amount
            }
        
        # PRIORITY 2: Mute commands
        if any(word in text for word in ["мут", "мьют", "заглушить"]):
            channel = self._extract_channel_number(text)
            if channel is not None:
                logger.info(f"✅ Parsed mute_channel command: channel={channel}")
                return {
                    "type": "mute_channel",
                    "channel": channel,
                    "muted": True
                }
        
        # PRIORITY 3: Load snapshot
        if any(word in text for word in ["загрузить", "снапшот", "сцена"]):
            # Extract name after command word
            for keyword in ["загрузить", "снапшот", "сцена"]:
                if keyword in text:
                    parts = text.split(keyword)
                    if len(parts) > 1:
                        snap_name = parts[1].strip()
                        if snap_name:
                            logger.info(f"✅ Parsed load_snap command: snap_name={snap_name}")
                            return {
                                "type": "load_snap",
                                "snap_name": snap_name
                            }
        
        # PRIORITY 4: Gain commands
        if "гейн" in text:
            channel = self._extract_channel_number(text)
            if channel is not None:
                logger.info(f"✅ Parsed set_gain command: channel={channel}")
                return {
                    "type": "set_gain",
                    "channel": channel,
                    "value": 0.0
                }
        
        # PRIORITY 5: Channel/Fader selection (only if no volume adjustment)
        if "канал" in text or "фейдер" in text or "channel" in text:
            channel = self._extract_channel_number(text)
            if channel is not None:
                logger.info(f"✅ Parsed set_fader command: channel={channel}")
                return {
                    "type": "set_fader",
                    "channel": channel,
                    "value": 0.5
                }
        
        # If no specific command found
        logger.warning(f"❌ No command pattern matched for: {text}")
        return None
    
    def add_command(self, keyword: str, pattern: str, action: str):
        """
        Add custom command mapping
        
        Args:
            keyword: Keyword to trigger command recognition
            pattern: Regex pattern to extract parameters
            action: Action type name
        """
        self.command_map[keyword] = {
            "pattern": pattern,
            "action": action
        }
        logger.info(f"Added custom command: {keyword} -> {action}")
