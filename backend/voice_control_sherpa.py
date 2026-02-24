"""
Voice Control Module using Sherpa-ONNX with GigaAM Russian Model
Optimized for Russian voice commands - much better than Whisper for Russian
"""
import logging
import pyaudio
import numpy as np
import re
import time
import os
from typing import Optional, Callable, Dict, Any
import threading
import queue

try:
    import sherpa_onnx
    SHERPA_AVAILABLE = True
except ImportError:
    SHERPA_AVAILABLE = False
    sherpa_onnx = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    logger.warning("webrtcvad not available. Install with: pip install webrtcvad")

from rapidfuzz import fuzz


class VoiceControlSherpa:
    """
    Voice control using Sherpa-ONNX with GigaAM Russian model
    Much more accurate for Russian than Whisper
    """
    
    # Command vocabulary - words to keep during correction
    COMMAND_VOCAB = {
        # Commands
        "тише", "громче", "потише", "погромче", "еще", "ещё",
        "минус", "плюс", "убрать", "убавь", "прибавь", "добавь",
        "мут", "мьют", "замьютить", "заглушить", "включить", "выключить", "размьютить",
        "загрузить", "снапшот", "сцена", "пресет", "сделать", "поставить",
        "канал", "фейдер", "гейн", "соло",
        # Numbers
        "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять",
        "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
        "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
        "двадцать", "тридцать", "сорок",
        "первый", "второй", "третий", "четвёртый", "четвертый", "пятый", "шестой", 
        "седьмой", "восьмой", "девятый", "десятый",
        "децибел", "децибела", "дб", "на",
        # EQ and Compressor
        "эквалайзер", "эква", "eq", "частоты", "низкие", "высокие", "басы", "верхи", "низы",
        "компрессор", "комп", "компрессия", "динамика", "dyn", "сжатие",
        "порог", "threshold", "threshold", "ratio", "рейшн", "атака", "attack", "релиз", "release",
        "гейн", "gain", "микс", "mix", "колено", "knee",
        # Instruments - drums
        "бочка", "кик", "kick", "бас-барабан", "басбарабан", "bd",
        "малый", "малый барабан", "малыйбарабан", "рабочий", "снэйр", "снейр", "snare",
        "том", "томы", "средний", "высокий", "низкий",
        "флортом", "флор",
        "хайхэт", "хэт", "хай", "hihat",
        "райд", "ride",
        "оверхэд", "оверхэды", "overhead", "овер",
        "тарелки", "крэш", "crash",
        "барабаны", "ударные", "drums",
        # Instruments - other
        "бас", "басгитара", "bass",
        "гитара", "электрогитара", "акустика",
        "клавиши", "клавишные", "синт", "синтезатор", "пиано", "piano", "keys",
        "аккордеон", "гармошка", "баян",
        "скрипка", "violin",
        "труба", "саксофон", "духовые",
        # Vocals
        "вокал", "голос", "vocal", "микрофон", "мик",
        "бэквокал", "бэки", "подпевка",
        # Other sources
        "плейбэк", "playback", "минусовка", "фонограмма",
        "ди-ай", "di", "директ",
        "лайн", "line",
        # Names (common musician names)
        "слава", "сергей", "сережа", "дима", "руслан", "андрей", "саша", "александр",
        "миша", "михаил", "коля", "николай", "паша", "павел", "вова", "владимир",
        "женя", "евгений", "макс", "максим", "артем", "антон", "игорь", "олег",
    }
    
    # Instrument/Name aliases -> channel numbers (user configurable)
    # Default mapping - can be updated via set_channel_aliases()
    CHANNEL_ALIASES = {
        # Drums (typical setup channels 1-8)
        "бочка": 1, "бочку": 1, "бочке": 1, "бочки": 1,
        "кик": 1, "kick": 1, "бас-барабан": 1, "басбарабан": 1, "bd": 1,
        "малый": 2, "рабочий": 2, "снэйр": 2, "снейр": 2, "snare": 2,
        "хайхэт": 3, "хэт": 3, "хай": 3, "hihat": 3,
        "том": 4, "высокий том": 4,
        "средний том": 5, "средний": 5,
        "флортом": 6, "флор": 6, "низкий том": 6,
        "оверхэд": 7, "овер": 7, "overhead": 7, "оверхэды": 7, "тарелки": 7,
        # Bass
        "бас": 9, "басгитара": 9, "bass": 9,
        # Guitar
        "гитара": 10, "электрогитара": 10,
        # Keys
        "клавиши": 11, "клавишные": 11, "синт": 11, "пиано": 11, "keys": 11,
        "аккордеон": 12, "гармошка": 12, "баян": 12,
        # Vocals (typical channels 17-24)
        "вокал": 17, "голос": 17, "vocal": 17,
        "первый вокал": 17, "главный вокал": 17,
        "второй вокал": 18, "бэквокал": 18, "бэки": 18,
        "третий вокал": 19,
        # Playback
        "плейбэк": 25, "playback": 25, "минусовка": 25, "фонограмма": 25,
        # Common names (can be customized)
        "слава": None, "сергей": None, "сережа": None, "дима": None, "руслан": None,
    }
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 sample_rate: int = 16000,
                 input_device_index: Optional[int] = None,
                 input_channel: int = 0,
                 **kwargs):  # Accept extra kwargs for compatibility
        
        if not SHERPA_AVAILABLE:
            raise ImportError("sherpa-onnx is not installed. Please install with: pip install sherpa-onnx")
        
        self.sample_rate = sample_rate
        self.input_device_index = input_device_index
        self.input_channel = input_channel
        
        # Find model path
        if model_path is None:
            base_dir = os.path.dirname(__file__)
            possible_paths = [
                os.path.join(base_dir, "models", "sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24"),
                os.path.join(base_dir, "models", "giga-am-russian"),
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    model_path = p
                    break
        
        self.model_path = model_path
        self.recognizer = None
        self.stream = None
        self.audio = None
        self.audio_stream = None
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.command_callback: Optional[Callable[[Dict], None]] = None
        self.processing_thread: Optional[threading.Thread] = None
        
        # Copy class-level aliases to instance (so each instance can have custom mapping)
        self.channel_aliases = dict(self.CHANNEL_ALIASES)
        
        # Remember last channel for "еще" commands
        self.last_channel: Optional[int] = None
        
        # Initialize VAD if available
        self.vad = None
        if VAD_AVAILABLE:
            try:
                # VAD mode: 0=quality, 1=low bitrate, 2=aggressive, 3=very aggressive
                # Mode 1 works better - less aggressive, catches more speech
                self.vad = webrtcvad.Vad(1)  # Low bitrate mode - better for speech detection
                logger.info("VAD initialized (mode 1 - low bitrate)")
            except Exception as e:
                logger.warning(f"Failed to initialize VAD: {e}")
        
        logger.info(f"VoiceControlSherpa initialized (model: {model_path})")
    
    def set_channel_aliases(self, aliases: Dict[str, int]):
        """
        Set custom channel aliases for instruments/names
        Example: {"бочка": 1, "дима": 17, "слава": 18}
        """
        self.channel_aliases.update(aliases)
        logger.info(f"Updated channel aliases: {aliases}")
    
    def get_channel_aliases(self) -> Dict[str, int]:
        """Get current channel aliases"""
        return self.channel_aliases
    
    def load_model(self):
        """Load Sherpa-ONNX model"""
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        logger.info("=" * 60)
        logger.info(f"LOADING SHERPA-ONNX GigaAM MODEL")
        logger.info(f"Path: {self.model_path}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Create recognizer config
        model_file = os.path.join(self.model_path, "model.int8.onnx")
        tokens_file = os.path.join(self.model_path, "tokens.txt")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        if not os.path.exists(tokens_file):
            raise FileNotFoundError(f"Tokens file not found: {tokens_file}")
        
        # Create OFFLINE recognizer (GigaAM requires this)
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=model_file,
            tokens=tokens_file,
            num_threads=4,
            sample_rate=self.sample_rate,
            feature_dim=64,  # GigaAM uses 64 mel bins
            decoding_method="greedy_search",
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✅ SHERPA-ONNX MODEL LOADED in {elapsed:.1f}s")
    
    def start_listening(self, callback: Callable[[Dict], None]):
        """Start listening for voice commands"""
        if self.is_listening:
            logger.warning("Already listening")
            return
        
        if not self.recognizer:
            self.load_model()
        
        self.command_callback = callback
        self.is_listening = True
        
        try:
            self.audio = pyaudio.PyAudio()
            
            # Open audio stream
            self.audio_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=int(self.sample_rate * 0.1),  # 100ms chunks
                stream_callback=self._audio_callback
            )
            
            self.audio_stream.start_stream()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._process_audio, daemon=True)
            self.processing_thread.start()
            
            logger.info("✅ Sherpa-ONNX voice control started listening")
            
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            self.is_listening = False
            raise
    
    def stop_listening(self):
        """Stop listening"""
        if not self.is_listening:
            return
        
        self.is_listening = False
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        logger.info("Sherpa-ONNX voice control stopped")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback"""
        if self.is_listening:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data)
        return (None, pyaudio.paContinue)
    
    def _process_audio(self):
        """Process audio with Sherpa-ONNX offline recognition using VAD"""
        logger.info("Sherpa-ONNX processing thread started")
        
        audio_buffer = []
        buffer_duration = 6.0  # seconds - longer buffer for complete phrases
        buffer_size = int(self.sample_rate * buffer_duration)
        last_process_time = time.time()
        last_speech_time = None
        silence_threshold = 0.8  # seconds of silence after speech before processing (reduced for faster response)
        min_speech_duration = 0.2  # minimum speech duration to process (reduced to catch short commands)
        
        # VAD frame size: 10ms, 20ms, or 30ms (webrtcvad requirement)
        vad_frame_size_ms = 30  # 30ms frames for 16kHz
        vad_frame_size = int(self.sample_rate * vad_frame_size_ms / 1000)
        
        while self.is_listening:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                audio_buffer.append(audio_data)
                
                current_time = time.time()
                total_samples = sum(len(chunk) for chunk in audio_buffer)
                time_since_last = current_time - last_process_time
                
                # Use VAD if available
                has_speech = False
                if self.vad and len(audio_buffer) > 0:
                    # Check last chunk for speech (VAD works on 16-bit PCM)
                    last_chunk = audio_buffer[-1]
                    if len(last_chunk) >= vad_frame_size:
                        # Convert float32 to int16 for VAD
                        int16_audio = (last_chunk[:vad_frame_size] * 32767).astype(np.int16)
                        try:
                            has_speech = self.vad.is_speech(int16_audio.tobytes(), self.sample_rate)
                            if has_speech:
                                if last_speech_time is None:
                                    logger.debug("VAD: Speech detected (start)")
                                last_speech_time = current_time
                        except Exception as e:
                            logger.debug(f"VAD error: {e}")
                
                # Determine if we should process
                # 1. Buffer is full
                # 2. VAD detected silence after speech (if VAD available)
                # 3. Time-based silence detection (fallback if no VAD)
                should_process = False
                
                if total_samples >= buffer_size:
                    should_process = True
                    logger.debug("Buffer full, processing...")
                elif self.vad and last_speech_time is not None:
                    # VAD detected speech before, now check for silence
                    silence_duration = current_time - last_speech_time
                    speech_duration = total_samples / self.sample_rate
                    if silence_duration >= silence_threshold and total_samples > self.sample_rate * min_speech_duration:
                        should_process = True
                        logger.info(f"VAD: {silence_duration:.2f}s silence after {speech_duration:.2f}s speech, processing...")
                    elif silence_duration >= silence_threshold:
                        logger.debug(f"VAD: {silence_duration:.2f}s silence but speech too short ({speech_duration:.2f}s < {min_speech_duration}s)")
                elif not self.vad:
                    # Fallback: time-based detection
                    if total_samples > self.sample_rate * min_speech_duration and time_since_last > silence_threshold:
                        should_process = True
                        logger.debug(f"Time-based: {time_since_last:.2f}s silence, processing...")
                
                if should_process and total_samples > 0:
                    # Concatenate audio
                    audio = np.concatenate(audio_buffer)
                    audio_buffer = []
                    last_process_time = current_time
                    last_speech_time = None  # Reset speech detection
                    
                    # Create stream and decode
                    stream = self.recognizer.create_stream()
                    stream.accept_waveform(self.sample_rate, audio)
                    self.recognizer.decode_stream(stream)
                    
                    text = stream.result.text.strip().lower()
                    
                    if text:
                        logger.info(f"🎤 Recognized: '{text}'")
                        
                        # Process command
                        corrected = self._correct_text(text)
                        if corrected != text:
                            logger.info(f"   Corrected: '{corrected}'")
                        
                        command = self._parse_command(corrected)
                        if command and self.command_callback:
                            try:
                                self.command_callback(command)
                            except Exception as e:
                                logger.error(f"Error in callback: {e}")
                
            except queue.Empty:
                # Process remaining buffer after silence (VAD or time-based)
                current_time = time.time()
                total_samples = sum(len(chunk) for chunk in audio_buffer) if audio_buffer else 0
                time_since_last = current_time - last_process_time
                
                # Check VAD silence or time-based
                should_process_empty = False
                if self.vad and last_speech_time is not None:
                    silence_duration = current_time - last_speech_time
                    if silence_duration >= silence_threshold and total_samples > self.sample_rate * min_speech_duration:
                        should_process_empty = True
                elif not self.vad:
                    if total_samples > self.sample_rate * min_speech_duration and time_since_last > silence_threshold:
                        should_process_empty = True
                
                if should_process_empty:
                    audio = np.concatenate(audio_buffer)
                    audio_buffer = []
                    last_process_time = current_time
                    last_speech_time = None  # Reset speech detection
                    
                    stream = self.recognizer.create_stream()
                    stream.accept_waveform(self.sample_rate, audio)
                    self.recognizer.decode_stream(stream)
                    
                    text = stream.result.text.strip().lower()
                    
                    if text:
                        logger.info(f"🎤 Recognized (after silence): '{text}'")
                        corrected = self._correct_text(text)
                        if corrected != text:
                            logger.info(f"   Corrected: '{corrected}'")
                        
                        command = self._parse_command(corrected)
                        if command and self.command_callback:
                            try:
                                self.command_callback(command)
                            except Exception as e:
                                logger.error(f"Error in callback: {e}")
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
        
        logger.info("Sherpa-ONNX processing thread stopped")
    
    def _correct_text(self, text: str) -> str:
        """Correct recognition errors and normalize text"""
        replacements = {
            # Symbols
            r"[–—−]": "минус",
            r"\s+-\s+": " минус ",
            r"[.!?,]": "",
            # Volume commands
            r"\bтишин\w*\b": "тише",
            r"\bтихо\b": "тише",
            r"\bтиши\b": "тише",
            r"\bпотише\b": "тише",
            r"\bгромко\b": "громче",
            r"\bгром\b": "громче",
            r"\bпогромче\b": "громче",
            r"\bубер\w*\b": "убрать",
            r"\bубав\w*\b": "убавь",
            r"\bприбав\w*\b": "прибавь",
            r"\bдобав\w*\b": "добавь",
            # Mute commands
            r"\bзамьютить\b": "мьют",
            r"\bзамутить\b": "мьют",
            r"\bмутить\b": "мьют",
            r"\bразмьютить\b": "размьютить",
            r"\bанмьют\b": "размьютить",
            # Channel/fader
            r"\bпервой\b": "первый",
            r"\bлевый\b": "первый",
            r"\bправый\b": "второй",
            r"\bканала?\b": "канал",
            r"\bканалов\b": "канал",
            r"\bфейдера?\b": "фейдер",
            # Decibels
            r"\bдецибел\w*\b": "децибела",
            r"\bдицибел\w*\b": "децибела",
            r"\bдб\b": "децибела",
            # Instruments corrections - common recognition errors
            r"\bхайхед\b": "хайхэт",
            r"\bхайхет\b": "хайхэт",
            r"\bоверхед\b": "оверхэд",
            r"\bоверхет\b": "оверхэд",
            r"\bбасбарабан\b": "бочка",
            r"\bкикдрам\b": "кик",
            r"\bснейяр\b": "снэйр",
            r"\bснеяр\b": "снэйр",
            r"\bгормошка\b": "гармошка",
            r"\bгормон\b": "гармошка",
            # Common bochka recognition errors
            r"\bбозбочк\w*\b": "бочка",
            r"\bбозочк\w*\b": "бочка",
            r"\bбочк\b": "бочка",
            r"\bбочку\b": "бочка",
            r"\bбочке\b": "бочка",
            r"\bбочки\b": "бочка",
            r"\bбасбочк\w*\b": "бочка",
            # Remove filler words
            r"\bи\b": "",
            r"\bна\s+на\b": "на",
        }
        
        result = text.lower()
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result)
        
        # Fuzzy match to vocabulary
        words = result.split()
        corrected_words = []
        
        for word in words:
            if word.isdigit():
                corrected_words.append(word)
                continue
            if len(word) < 2:
                continue
            
            if word in self.COMMAND_VOCAB:
                corrected_words.append(word)
            else:
                best_match = None
                best_score = 0
                for vocab_word in self.COMMAND_VOCAB:
                    score = fuzz.ratio(word, vocab_word)
                    if score > best_score and score >= 70:
                        best_score = score
                        best_match = vocab_word
                
                if best_match:
                    corrected_words.append(best_match)
                elif len(word) > 2:
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
        """Extract channel number from text - checks aliases first, then numbers"""
        words = text.split()
        db_words = {"минус", "плюс", "на"}
        channel_words = {"канал", "фейдер"}
        
        # 1. Check for instrument/name aliases FIRST (most specific)
        # Try multi-word phrases first (e.g. "первый вокал", "средний том")
        for phrase_len in [3, 2]:
            for i in range(len(words) - phrase_len + 1):
                phrase = " ".join(words[i:i+phrase_len])
                if phrase in self.channel_aliases:
                    ch = self.channel_aliases[phrase]
                    if ch is not None:
                        logger.info(f"   Alias match: '{phrase}' -> ch {ch}")
                        return ch
        
        # Try single words
        for word in words:
            if word in self.channel_aliases:
                ch = self.channel_aliases[word]
                if ch is not None:
                    logger.info(f"   Alias match: '{word}' -> ch {ch}")
                    return ch
        
        # 2. Find number after "канал"/"фейдер"
        for i, word in enumerate(words):
            if word in channel_words and i + 1 < len(words):
                next_word = words[i + 1]
                if next_word.isdigit():
                    return int(next_word)
                num = self._word_to_number(next_word)
                if num:
                    return num
        
        # 3. Find ordinals (первый, второй, etc.) followed by "канал" or standalone
        for i, word in enumerate(words):
            if word in ["первый", "второй", "третий", "четвёртый", "четвертый",
                       "пятый", "шестой", "седьмой", "восьмой", "девятый", "десятый",
                       "одиннадцатый", "двенадцатый"]:
                # Check if followed by "канал" (more specific)
                if i + 1 < len(words) and words[i + 1] in channel_words:
                    return self._word_to_number(word)
                # Check if followed by instrument name (e.g. "первый вокал")
                if i + 1 < len(words):
                    next_phrase = f"{word} {words[i + 1]}"
                    if next_phrase in self.channel_aliases:
                        continue  # Already handled above
                # Standalone ordinal
                return self._word_to_number(word)
        
        # 4. Find plain number not after db_words
        for i, word in enumerate(words):
            if i > 0 and words[i-1] in db_words:
                continue
            if word.isdigit():
                num = int(word)
                if 1 <= num <= 48:
                    return num
        
        return None
    
    def _extract_db(self, text: str) -> Optional[int]:
        """Extract dB amount"""
        text_lower = text.lower()
        
        # First check for numeric patterns: "на 5", "минус 3", "5 дб", "1дб"
        for pattern in [
            r'(?:на|минус|плюс)\s*(\d+)',
            r'(\d+)\s*(?:децибел|дб)',
            r'(\d+)(?:дб|децибел)'
        ]:
            match = re.search(pattern, text_lower)
            if match:
                return int(match.group(1))
        
        # Check for number words in text (order matters - check longer numbers first!)
        number_words = [
            ("двадцать", 20), ("тридцать", 30), ("сорок", 40),
            ("одиннадцать", 11), ("двенадцать", 12), 
            ("тринадцать", 13), ("четырнадцать", 14), ("пятнадцать", 15),
            ("шестнадцать", 16), ("семнадцать", 17), ("восемнадцать", 18), ("девятнадцать", 19),
            ("десять", 10), ("девять", 9), ("восемь", 8), ("семь", 7), ("шесть", 6),
            ("пять", 5), ("четыре", 4), ("три", 3), ("два", 2), ("один", 1)
        ]
        
        # First check for explicit patterns: "на число", "минус число", "число децибел"
        for word, value in number_words:
            # Pattern: "на/минус/плюс число" or "число децибел/дб"
            if re.search(rf'(?:на|минус|плюс)\s+{word}(?:\s+децибел|дб)?', text_lower) or \
               re.search(rf'{word}\s+(?:децибел|дб)', text_lower):
                return value
        
        # Then check if number word appears as standalone word (not part of another word)
        # Use word boundaries to avoid matching "один" in "одиннадцать"
        for word, value in number_words:
            # Match as whole word (with word boundaries)
            if re.search(rf'\b{word}\b', text_lower):
                return value
        
        return None
    
    def _parse_command(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse text into command"""
        if not text or len(text) < 2:
            return None
        
        garbage = {"субтитры", "пиши", "старый", "пусть", "а", "но", "то", "ну", "эээ", "ммм"}
        if set(text.split()).issubset(garbage):
            return None
        
        logger.info(f"Parsing: '{text}'")
        
        # Check for "еще" commands - use last channel if no channel specified
        has_esche = any(w in text for w in ["еще", "ещё"])
        
        # EQ band words - if present, skip volume handling and let EQ handle it
        eq_band_words = ["верхи", "низы", "басы", "низкие", "высокие", "середина", "средние"]
        has_eq_band = any(w in text for w in eq_band_words)
        
        # Volume DOWN - check first as most common
        # Skip if EQ band words present - let EQ handler process these
        if any(w in text for w in ["тише", "минус", "убрать", "убавь"]) and not has_eq_band:
            channel = self._extract_channel(text)
            db = self._extract_db(text)
            
            # If no channel found, try to use last channel (for "еще" or broken phrases)
            if channel is None and self.last_channel is not None:
                # Check if this looks like a continuation command
                words_only = set(text.split())
                continuation_words = {"тише", "потише", "еще", "ещё", "на", "децибела", "децибел", "дб", 
                                     "минус", "плюс", "один", "два", "три", "четыре", "пять", "шесть", 
                                     "семь", "восемь", "девять", "десять"}
                # Use last channel if: has "еще", or only continuation words, or has explicit dB value
                if has_esche or words_only.issubset(continuation_words) or db is not None:
                    channel = self.last_channel
                    logger.info(f"   Using last channel {channel} (no channel in command)")
            
            db = db or 3
            logger.info(f"✅ volume_down: ch={channel}, db={db}")
            result = {"type": "volume_down", "channel": channel, "db": db, "amount": db/100}
            # Remember channel for next "еще" command
            if channel:
                self.last_channel = channel
            return result
        
        # Volume UP
        # Skip if EQ band words present - let EQ handler process these
        if any(w in text for w in ["громче", "плюс", "прибавь", "добавь"]) and not has_eq_band:
            channel = self._extract_channel(text)
            db = self._extract_db(text)
            
            # If no channel found, try to use last channel (for "еще" or broken phrases)
            if channel is None and self.last_channel is not None:
                # Check if this looks like a continuation command
                words_only = set(text.split())
                continuation_words = {"громче", "погромче", "еще", "ещё", "на", "децибела", "децибел", "дб",
                                     "минус", "плюс", "один", "два", "три", "четыре", "пять", "шесть",
                                     "семь", "восемь", "девять", "десять"}
                # Use last channel if: has "еще", or only continuation words, or has explicit dB value
                if has_esche or words_only.issubset(continuation_words) or db is not None:
                    channel = self.last_channel
                    logger.info(f"   Using last channel {channel} (no channel in command)")
            
            db = db or 3
            logger.info(f"✅ volume_up: ch={channel}, db={db}")
            result = {"type": "volume_up", "channel": channel, "db": db, "amount": db/100}
            # Remember channel for next "еще" command
            if channel:
                self.last_channel = channel
            return result
        
        # EQ commands - MUST BE BEFORE Unmute to catch "включить эквалайзер"
        if any(w in text for w in ["эквалайзер", "эква", "eq", "частоты", "низкие", "высокие", "басы", "верхи", "низы"]):
            channel = self._extract_channel(text) or self.last_channel
            if channel:
                self.last_channel = channel
                # Extract band and action
                band = None
                if any(w in text for w in ["низкие", "басы", "низы", "low"]):
                    band = "low"
                elif any(w in text for w in ["высокие", "верхи", "high"]):
                    band = "high"
                elif any(w in text for w in ["середина", "средние", "mid"]):
                    band = "mid"
                elif any(w in text for w in ["первая", "полоса 1"]):
                    band = "1"
                elif any(w in text for w in ["вторая", "полоса 2"]):
                    band = "2"
                elif any(w in text for w in ["третья", "полоса 3"]):
                    band = "3"
                elif any(w in text for w in ["четвертая", "полоса 4"]):
                    band = "4"
                
                # Extract gain change - check for "добавь", "убрать" etc even without explicit dB
                if any(w in text for w in ["громче", "плюс", "прибавь", "добавь", "поднять", "больше"]):
                    db = self._extract_db(text) or 3  # Default 3 dB if not specified
                    logger.info(f"✅ eq_band_up: ch={channel}, band={band}, db={db}")
                    return {"type": "eq_band_up", "channel": channel, "band": band, "db": db}
                elif any(w in text for w in ["тише", "минус", "убрать", "убавь", "опустить", "меньше", "вырезать"]):
                    db = self._extract_db(text) or 3  # Default 3 dB if not specified
                    logger.info(f"✅ eq_band_down: ch={channel}, band={band}, db={db}")
                    return {"type": "eq_band_down", "channel": channel, "band": band, "db": db}
                elif any(w in text for w in ["включить", "включи"]):
                    logger.info(f"✅ eq_on: ch={channel}")
                    return {"type": "eq_on", "channel": channel, "on": 1}
                elif any(w in text for w in ["выключить", "выключи", "отключить"]):
                    logger.info(f"✅ eq_off: ch={channel}")
                    return {"type": "eq_on", "channel": channel, "on": 0}
        
        # Compressor commands - MUST BE BEFORE Unmute to catch "включить компрессор"
        if any(w in text for w in ["компрессор", "комп", "компрессия", "динамика", "dyn", "сжатие", "сжать"]):
            channel = self._extract_channel(text) or self.last_channel
            if channel:
                self.last_channel = channel
                # Extract parameter
                if any(w in text for w in ["порог", "threshold"]):
                    db = self._extract_db(text)
                    if db is not None:
                        # Negative threshold
                        threshold = -abs(db)
                        logger.info(f"✅ compressor_threshold: ch={channel}, threshold={threshold}")
                        return {"type": "compressor_threshold", "channel": channel, "threshold": threshold}
                elif any(w in text for w in ["гейн", "gain", "компенсация"]):
                    db = self._extract_db(text) or 3
                    if any(w in text for w in ["громче", "плюс", "прибавь", "добавь"]):
                        logger.info(f"✅ compressor_gain_up: ch={channel}, db={db}")
                        return {"type": "compressor_gain_up", "channel": channel, "db": db}
                    elif any(w in text for w in ["тише", "минус", "убрать", "убавь"]):
                        logger.info(f"✅ compressor_gain_down: ch={channel}, db={db}")
                        return {"type": "compressor_gain_down", "channel": channel, "db": db}
                elif any(w in text for w in ["сильнее", "больше", "жестче"]):
                    # Increase compression (lower threshold)
                    logger.info(f"✅ compressor_more: ch={channel}")
                    return {"type": "compressor_threshold", "channel": channel, "threshold": -3}
                elif any(w in text for w in ["слабее", "меньше", "мягче", "отпустить"]):
                    # Decrease compression (higher threshold)
                    logger.info(f"✅ compressor_less: ch={channel}")
                    return {"type": "compressor_threshold", "channel": channel, "threshold": 3}
                elif any(w in text for w in ["включить", "включи"]):
                    logger.info(f"✅ compressor_on: ch={channel}")
                    return {"type": "compressor_on", "channel": channel, "on": 1}
                elif any(w in text for w in ["выключить", "выключи", "отключить"]):
                    logger.info(f"✅ compressor_off: ch={channel}")
                    return {"type": "compressor_on", "channel": channel, "on": 0}
        
        # Mute
        if any(w in text for w in ["мут", "мьют", "заглушить", "замьютить"]):
            channel = self._extract_channel(text)
            if channel:
                logger.info(f"✅ mute: ch={channel}")
                self.last_channel = channel  # Remember channel
                return {"type": "mute_channel", "channel": channel, "muted": True}
            # Check if instrument/name without channel word
            for word in text.split():
                if word in self.channel_aliases and self.channel_aliases[word]:
                    ch = self.channel_aliases[word]
                    logger.info(f"✅ mute (by alias): {word} -> ch={ch}")
                    self.last_channel = ch  # Remember channel
                    return {"type": "mute_channel", "channel": ch, "muted": True}
        
        # Unmute - check AFTER EQ and Compressor to avoid catching "включить эквалайзер"
        if any(w in text for w in ["размьютить", "включить", "анмьют"]):
            # Skip if text contains EQ or compressor keywords (already handled above)
            if not any(w in text for w in ["эквалайзер", "эква", "eq", "компрессор", "комп", "компрессия", "динамика", "сжатие"]):
                channel = self._extract_channel(text)
                if channel:
                    logger.info(f"✅ unmute: ch={channel}")
                    self.last_channel = channel  # Remember channel
                    return {"type": "mute_channel", "channel": channel, "muted": False}
                for word in text.split():
                    if word in self.channel_aliases and self.channel_aliases[word]:
                        ch = self.channel_aliases[word]
                        logger.info(f"✅ unmute (by alias): {word} -> ch={ch}")
                        self.last_channel = ch  # Remember channel
                        return {"type": "mute_channel", "channel": ch, "muted": False}
        
        # Solo
        if "соло" in text:
            channel = self._extract_channel(text)
            if channel:
                logger.info(f"✅ solo: ch={channel}")
                self.last_channel = channel  # Remember channel
                return {"type": "solo_channel", "channel": channel, "solo": True}
        
        # Load snapshot
        if any(w in text for w in ["загрузить", "снапшот", "сцена", "пресет"]):
            for keyword in ["загрузить", "снапшот", "сцена", "пресет"]:
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
                self.last_channel = channel  # Remember channel
                db = self._extract_db(text) or 0
                # Check if up or down
                if any(w in text for w in ["громче", "плюс", "прибавь", "добавь"]):
                    logger.info(f"✅ gain_up: ch={channel}, db={db}")
                    return {"type": "gain_up", "channel": channel, "db": db}
                elif any(w in text for w in ["тише", "минус", "убрать", "убавь"]):
                    logger.info(f"✅ gain_down: ch={channel}, db={db}")
                    return {"type": "gain_down", "channel": channel, "db": db}
                else:
                    logger.info(f"✅ set_gain: ch={channel}")
                    return {"type": "set_gain", "channel": channel, "value": 0}
        
        # Check for standalone instrument/name with action keyword implied
        # e.g. "бочка тише" should work even without explicit channel word
        words = text.split()
        for word in words:
            if word in self.channel_aliases and self.channel_aliases[word]:
                ch = self.channel_aliases[word]
                # Check if any action follows
                remaining = text.split(word, 1)[-1] if word in text else ""
                if any(w in remaining for w in ["тише", "минус", "убрать", "убавь"]):
                    db = self._extract_db(text) or 3
                    logger.info(f"✅ volume_down (by alias): {word} -> ch={ch}, db={db}")
                    self.last_channel = ch  # Remember channel
                    return {"type": "volume_down", "channel": ch, "db": db, "amount": db/100}
                if any(w in remaining for w in ["громче", "плюс", "прибавь", "добавь"]):
                    db = self._extract_db(text) or 3
                    logger.info(f"✅ volume_up (by alias): {word} -> ch={ch}, db={db}")
                    self.last_channel = ch  # Remember channel
                    return {"type": "volume_up", "channel": ch, "db": db, "amount": db/100}
        
        # Channel/fader selection (less specific, check last)
        if "канал" in text or "фейдер" in text:
            channel = self._extract_channel(text)
            if channel:
                logger.info(f"✅ set_fader: ch={channel}")
                self.last_channel = channel  # Remember channel
                return {"type": "set_fader", "channel": channel, "value": 0.5}
        
        logger.debug(f"❌ No match: '{text}'")
        return None
