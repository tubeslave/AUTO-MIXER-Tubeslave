"""
Voice Control Module V2 for Auto Mixer
Optimized for Russian voice commands with multiple STT backends
"""
import asyncio
import logging
import pyaudio
import numpy as np
import re
import time
from typing import Optional, Callable, Dict, Any, List
from faster_whisper import WhisperModel
import threading
import queue
from rapidfuzz import fuzz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceControlV2:
    """
    Enhanced voice control with better Russian support
    Features:
    - Whisper with optimized settings for commands
    - Fuzzy matching for error correction
    - Command-focused recognition
    """
    
    # Command vocabulary for better recognition
    COMMAND_VOCAB = {
        # Actions
        "тише", "громче", "минус", "плюс", "убрать", "убавь", "прибавь", "добавь",
        "мут", "мьют", "заглушить", "включить", "выключить",
        "загрузить", "снапшот", "сцена", "пресет",
        "канал", "фейдер", "гейн",
        # Numbers
        "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять",
        "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
        "двадцать", "тридцать", "сорок",
        "первый", "второй", "третий", "четвёртый", "пятый", "шестой", "седьмой", "восьмой",
        # Units
        "децибел", "децибела", "дб",
        # Prepositions
        "на",
    }
    
    def __init__(self,
                 model_size: str = "small",
                 language: str = "ru",
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 input_device_index: Optional[int] = None,
                 input_channel: int = 0):
        
        self.model_size = model_size
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
        self.command_callback: Optional[Callable[[Dict], None]] = None
        self.processing_thread: Optional[threading.Thread] = None
        
        logger.info(f"VoiceControlV2 initialized (model: {model_size}, language: {language})")
    
    def load_model(self):
        """Load Whisper model with optimal settings"""
        logger.info("=" * 60)
        logger.info(f"LOADING WHISPER MODEL: {self.model_size}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Use int8 for faster CPU inference
        compute_type = "int8" if self.model_size in ["medium", "large", "large-v2", "large-v3"] else "default"
        
        self.model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=4
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✅ MODEL LOADED in {elapsed:.1f}s")
    
    def start_listening(self, callback: Callable[[Dict], None]):
        """Start listening for voice commands"""
        if self.is_listening:
            logger.warning("Already listening")
            return
        
        if not self.model:
            self.load_model()
        
        self.command_callback = callback
        self.is_listening = True
        
        try:
            self.audio = pyaudio.PyAudio()
            
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.stream.start_stream()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._process_audio, daemon=True)
            self.processing_thread.start()
            
            logger.info("✅ Voice control V2 started listening")
            
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            self.is_listening = False
            raise
    
    def stop_listening(self):
        """Stop listening"""
        if not self.is_listening:
            return
        
        self.is_listening = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        logger.info("Voice control V2 stopped")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback"""
        if self.is_listening:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def _process_audio(self):
        """Process audio and recognize speech"""
        audio_buffer = []
        buffer_duration = 3.0  # Shorter buffer for faster response
        buffer_size = int(self.sample_rate * buffer_duration * 2)
        last_process_time = time.time()
        
        while self.is_listening:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer.append(chunk)
                
                buffer_bytes = len(b''.join(audio_buffer))
                current_time = time.time()
                
                # Process every 2 seconds or when buffer is full
                should_process = (
                    buffer_bytes >= buffer_size or 
                    (buffer_bytes > self.sample_rate and current_time - last_process_time > 2.0)
                )
                
                if should_process and buffer_bytes > 0:
                    audio_data = np.frombuffer(b''.join(audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0
                    audio_buffer = []
                    last_process_time = current_time
                    
                    # Transcribe with command-optimized settings
                    segments, info = self.model.transcribe(
                        audio_data,
                        language=self.language,
                        beam_size=3,  # Reduced for speed
                        best_of=3,
                        temperature=0.0,
                        vad_filter=True,
                        vad_parameters=dict(
                            min_silence_duration_ms=500,
                            threshold=0.5,
                            speech_pad_ms=200
                        ),
                        no_speech_threshold=0.5,
                        log_prob_threshold=-0.8,  # Stricter
                        compression_ratio_threshold=2.0,
                        condition_on_previous_text=False,  # Important for short commands
                        without_timestamps=True,  # Faster
                    )
                    
                    for segment in segments:
                        text = segment.text.strip().lower()
                        
                        # Filter by confidence
                        if not text or len(text) < 2:
                            continue
                        if segment.no_speech_prob > 0.4:
                            continue
                        if segment.avg_logprob < -1.0:  # Low confidence
                            continue
                        
                        logger.info(f"🎤 Raw: '{text}' (conf: {segment.avg_logprob:.2f})")
                        
                        # Correct and parse
                        corrected = self._correct_text(text)
                        if corrected != text:
                            logger.info(f"   Corrected: '{corrected}'")
                        
                        command = self._parse_command(corrected)
                        if command and self.command_callback:
                            self.command_callback(command)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
    
    def _correct_text(self, text: str) -> str:
        """Correct recognition errors with fuzzy matching"""
        # Direct replacements
        replacements = {
            r"[–—−]": "минус",
            r"\s+-\s+": " минус ",
            r"\bтишин\w*\b": "тише",
            r"\bтихо\b": "тише",
            r"\bтиши\b": "тише",
            r"\bгромко\b": "громче",
            r"\bгром\b": "громче",
            r"\bубер\w*\b": "убрать",
            r"\bубав\w*\b": "убрать",
            r"\bприбав\w*\b": "прибавь",
            r"\bдобав\w*\b": "прибавь",
            r"\bпервой\b": "первый",
            r"\bканала?\b": "канал",
            r"\bдецибел\w*\b": "децибела",
            r"\bдб\b": "децибела",
            r"3d\b": "три децибела",
            # Remove garbage
            r"\bсубтитры\b": "",
            r"\bпиши\b": "",
            r"\bстарый\b": "",
            r"\bпусть\b": "",
            r"[.!?,]": "",
        }
        
        result = text.lower()
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result)
        
        # Fuzzy match words to vocabulary
        words = result.split()
        corrected_words = []
        
        for word in words:
            # Always keep numbers (digits), even single digit
            if word.isdigit():
                corrected_words.append(word)
                continue
            
            if len(word) < 2:
                continue
            
            # Check if word is in vocabulary or close to it
            if word in self.COMMAND_VOCAB:
                corrected_words.append(word)
            else:
                # Find closest match
                best_match = None
                best_score = 0
                for vocab_word in self.COMMAND_VOCAB:
                    score = fuzz.ratio(word, vocab_word)
                    if score > best_score and score >= 70:
                        best_score = score
                        best_match = vocab_word
                
                if best_match:
                    corrected_words.append(best_match)
                elif len(word) > 2:  # Keep unknown but meaningful words
                    corrected_words.append(word)
        
        return ' '.join(corrected_words).strip()
    
    def _word_to_number(self, word: str) -> Optional[int]:
        """Convert Russian number word to integer"""
        numbers = {
            "один": 1, "первый": 1,
            "два": 2, "второй": 2,
            "три": 3, "третий": 3,
            "четыре": 4, "четвёртый": 4, "четвертый": 4,
            "пять": 5, "пятый": 5,
            "шесть": 6, "шестой": 6,
            "семь": 7, "седьмой": 7,
            "восемь": 8, "восьмой": 8,
            "девять": 9, "девятый": 9,
            "десять": 10, "десятый": 10,
            "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
            "четырнадцать": 14, "пятнадцать": 15,
            "двадцать": 20, "тридцать": 30, "сорок": 40,
        }
        return numbers.get(word.lower())
    
    def _extract_channel(self, text: str) -> Optional[int]:
        """Extract channel number from text (not dB amount)"""
        words = text.split()
        
        # Words that indicate next number is dB, not channel
        db_indicator_words = {"минус", "плюс", "на"}
        # Words that indicate next number is channel
        channel_indicator_words = {"канал", "фейдер"}
        
        # First pass: find number after "канал" or "фейдер"
        for i, word in enumerate(words):
            if word in channel_indicator_words and i + 1 < len(words):
                next_word = words[i + 1]
                if next_word.isdigit():
                    return int(next_word)
                num = self._word_to_number(next_word)
                if num:
                    return num
        
        # Second pass: find ordinal (первый, второй...) anywhere
        for word in words:
            if word in ["первый", "второй", "третий", "четвёртый", "четвертый",
                       "пятый", "шестой", "седьмой", "восьмой", "девятый", "десятый"]:
                return self._word_to_number(word)
        
        # Third pass: find number that's NOT after db_indicator_words
        for i, word in enumerate(words):
            if i > 0 and words[i-1] in db_indicator_words:
                continue  # This is dB value, not channel
            
            if word.isdigit():
                num = int(word)
                if 1 <= num <= 40:
                    return num
        
        return None
    
    def _extract_db(self, text: str) -> Optional[int]:
        """Extract dB amount from text"""
        # Pattern: "на X", "минус X", "плюс X", "X децибела"
        for pattern in [
            r'(?:на|минус|плюс)\s*(\d+)',  # Allow no space
            r'(\d+)\s*(?:децибел|дб)',
        ]:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
        
        # Number words (check longer words first!)
        for word, value in [("двадцать", 20), ("тридцать", 30), ("десять", 10),
                           ("девять", 9), ("восемь", 8), ("семь", 7), ("шесть", 6),
                           ("пять", 5), ("четыре", 4), ("три", 3), ("два", 2), ("один", 1)]:
            if re.search(rf'(?:на|минус|плюс)\s*{word}', text):
                return value
            if re.search(rf'{word}\s*(?:децибел|дб)', text):
                return value
        
        # Also check for standalone numbers in text
        for word, value in [("двадцать", 20), ("тридцать", 30), ("десять", 10),
                           ("пять", 5), ("три", 3)]:
            if word in text:
                return value
        
        return None
    
    def _parse_command(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse corrected text into command"""
        if not text or len(text) < 2:
            return None
        
        # Garbage filter
        garbage = {"субтитры", "пиши", "старый", "пусть", "и", "а", "но", "то", "ну"}
        words = set(text.split())
        if words.issubset(garbage):
            return None
        
        logger.info(f"Parsing: '{text}'")
        
        # Volume DOWN
        if any(w in text for w in ["тише", "минус", "убрать", "убавь"]):
            channel = self._extract_channel(text)
            db = self._extract_db(text) or 3
            logger.info(f"✅ volume_down: ch={channel}, db={db}")
            return {"type": "volume_down", "channel": channel, "db": db, "amount": db/100}
        
        # Volume UP
        if any(w in text for w in ["громче", "плюс", "прибавь", "добавь"]):
            channel = self._extract_channel(text)
            db = self._extract_db(text) or 3
            logger.info(f"✅ volume_up: ch={channel}, db={db}")
            return {"type": "volume_up", "channel": channel, "db": db, "amount": db/100}
        
        # Mute
        if any(w in text for w in ["мут", "мьют", "заглушить"]):
            channel = self._extract_channel(text)
            if channel:
                logger.info(f"✅ mute: ch={channel}")
                return {"type": "mute_channel", "channel": channel, "muted": True}
        
        # Load snapshot
        if any(w in text for w in ["загрузить", "снапшот", "сцена"]):
            for keyword in ["загрузить", "снапшот", "сцена"]:
                if keyword in text:
                    parts = text.split(keyword)
                    if len(parts) > 1 and parts[1].strip():
                        name = parts[1].strip()
                        logger.info(f"✅ load_snap: {name}")
                        return {"type": "load_snap", "snap_name": name}
        
        # Gain
        if "гейн" in text:
            channel = self._extract_channel(text)
            if channel:
                logger.info(f"✅ set_gain: ch={channel}")
                return {"type": "set_gain", "channel": channel, "value": 0}
        
        # Channel/fader selection (lowest priority)
        if "канал" in text or "фейдер" in text:
            channel = self._extract_channel(text)
            if channel:
                logger.info(f"✅ set_fader: ch={channel}")
                return {"type": "set_fader", "channel": channel, "value": 0.5}
        
        logger.debug(f"❌ No match: '{text}'")
        return None
