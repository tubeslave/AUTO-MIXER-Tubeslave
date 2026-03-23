"""
Voice Control Module using Vosk
Optimized for Russian voice commands with limited vocabulary
"""
import asyncio
import logging
import json
import os
import queue
import threading
import time
import re
from typing import Optional, Callable, Dict, Any

import pyaudio
from vosk import Model, KaldiRecognizer, SetLogLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Vosk debug output
SetLogLevel(-1)


class VoiceControlVosk:
    """
    Voice control handler using Vosk STT
    Optimized for Russian mixer commands with grammar-based recognition
    """
    
    # Define the grammar (allowed words and phrases)
    GRAMMAR = [
        # Numbers
        "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять",
        "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
        "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать", "двадцать",
        "тридцать", "сорок",
        # Ordinals
        "первый", "второй", "третий", "четвёртый", "пятый", "шестой", "седьмой", "восьмой", "девятый", "десятый",
        # Commands
        "канал", "фейдер", "гейн", "громкость",
        "тише", "громче", "убавь", "прибавь", "убрать", "добавить",
        "минус", "плюс", "на",
        "мут", "мьют", "заглушить", "включить", "выключить",
        "загрузить", "снапшот", "сцена", "пресет",
        "децибел", "децибела", "дб",
        # Filler words that might appear
        "сделай", "сделать", "поставь", "установи", "выстави"
    ]
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 sample_rate: int = 16000,
                 input_device_index: Optional[int] = None,
                 input_channel: int = 0,
                 use_grammar: bool = True):
        """
        Initialize Vosk voice control
        
        Args:
            model_path: Path to Vosk model directory
            sample_rate: Audio sample rate (Hz)
            input_device_index: Audio input device index
            input_channel: Audio input channel
            use_grammar: Whether to use limited grammar for better accuracy
        """
        self.sample_rate = sample_rate
        self.input_device_index = input_device_index
        self.input_channel = input_channel
        self.use_grammar = use_grammar
        
        # Find model path
        if model_path is None:
            # Look for model in standard locations
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "models", "vosk-model-small-ru-0.22"),
                os.path.join(os.path.dirname(__file__), "models", "vosk-model-small-ru"),
                os.path.expanduser("~/.vosk/vosk-model-small-ru-0.22"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        self.model_path = model_path
        self.model: Optional[Model] = None
        self.recognizer: Optional[KaldiRecognizer] = None
        
        self.audio = None
        self.stream = None
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.command_callback: Optional[Callable[[Dict], None]] = None
        self.processing_thread: Optional[threading.Thread] = None
        
        logger.info(f"VoiceControlVosk initialized (model_path: {model_path})")
    
    def load_model(self):
        """Load Vosk model"""
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Vosk model not found at {self.model_path}")
        
        logger.info("=" * 60)
        logger.info(f"LOADING VOSK MODEL from {self.model_path}")
        logger.info("=" * 60)
        
        start_time = time.time()
        self.model = Model(self.model_path)
        
        # Create recognizer with or without grammar
        if self.use_grammar:
            grammar_json = json.dumps(self.GRAMMAR, ensure_ascii=False)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate, grammar_json)
        else:
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        
        self.recognizer.SetWords(True)
        
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"✅ VOSK MODEL LOADED in {elapsed:.2f} seconds")
        logger.info("=" * 60)
    
    def start_listening(self, callback: Callable[[Dict], None]):
        """Start listening for voice commands"""
        logger.info("=" * 60)
        logger.info("START_LISTENING CALLED (Vosk)")
        logger.info("=" * 60)
        
        if self.is_listening:
            logger.warning("Already listening")
            return
        
        if not self.model:
            logger.info("Loading Vosk model...")
            self.load_model()
        
        self.command_callback = callback
        self.is_listening = True
        
        try:
            self.audio = pyaudio.PyAudio()
            
            # Get device info
            if self.input_device_index is not None:
                device_info = self.audio.get_device_info_by_index(self.input_device_index)
                logger.info(f"Using audio device: {device_info['name']}")
            
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=4000,  # Larger buffer for Vosk
                stream_callback=self._audio_callback
            )
            
            self.stream.start_stream()
            logger.info(f"Audio stream started on device {self.input_device_index}")
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._process_audio, daemon=True)
            self.processing_thread.start()
            
            logger.info("✅ Vosk voice control started listening")
            
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}", exc_info=True)
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
        
        logger.info("Vosk voice control stopped listening")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if self.is_listening:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def _process_audio(self):
        """Process audio and recognize speech using Vosk"""
        logger.info("Vosk audio processing thread started")
        
        while self.is_listening:
            try:
                data = self.audio_queue.get(timeout=0.1)
                
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip()
                    
                    if text:
                        logger.info(f"🎤 Vosk recognized: '{text}'")
                        
                        # Parse and execute command
                        command = self._parse_command(text)
                        if command and self.command_callback:
                            try:
                                self.command_callback(command)
                            except Exception as e:
                                logger.error(f"Error in command callback: {e}", exc_info=True)
                else:
                    # Partial result (for real-time feedback if needed)
                    partial = json.loads(self.recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip()
                    if partial_text and len(partial_text) > 3:
                        logger.debug(f"Partial: {partial_text}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audio: {e}", exc_info=True)
        
        logger.info("Vosk audio processing thread stopped")
    
    def _word_to_number(self, word: str) -> Optional[int]:
        """Convert Russian number words to integers"""
        number_map = {
            "один": 1, "два": 2, "три": 3, "четыре": 4, "пять": 5,
            "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
            "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14,
            "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18,
            "девятнадцать": 19, "двадцать": 20, "тридцать": 30, "сорок": 40,
            # Ordinals
            "первый": 1, "второй": 2, "третий": 3, "четвёртый": 4, "четвертый": 4,
            "пятый": 5, "шестой": 6, "седьмой": 7, "восьмой": 8, "девятый": 9, "десятый": 10,
        }
        return number_map.get(word.lower())
    
    def _extract_channel_number(self, text: str) -> Optional[int]:
        """Extract channel number from text"""
        words = text.lower().split()
        
        # First look for ordinals before "канал"
        for i, word in enumerate(words):
            num = self._word_to_number(word)
            if num and num <= 40:
                # Check if this is likely a channel number (near "канал" or ordinal)
                if i + 1 < len(words) and words[i + 1] == "канал":
                    return num
                if i > 0 and words[i - 1] == "канал":
                    return num
                # If it's an ordinal (первый, второй...), use it
                if word in ["первый", "второй", "третий", "четвёртый", "четвертый", 
                           "пятый", "шестой", "седьмой", "восьмой", "девятый", "десятый"]:
                    return num
        
        # Then look for digit after "канал"
        match = re.search(r'канал\s+(\d+)', text)
        if match:
            return int(match.group(1))
        
        # Or digit before "канал"
        match = re.search(r'(\d+)\s+канал', text)
        if match:
            return int(match.group(1))
        
        # Standalone ordinal
        for word in words:
            if word in ["первый", "второй", "третий", "четвёртый", "четвертый",
                       "пятый", "шестой", "седьмой", "восьмой", "девятый", "десятый"]:
                return self._word_to_number(word)
        
        return None
    
    def _extract_db_amount(self, text: str) -> Optional[float]:
        """Extract decibel amount from text"""
        words = text.lower().split()
        
        # Look for patterns like "минус три", "на пять", "плюс десять"
        for i, word in enumerate(words):
            if word in ["минус", "плюс", "на"]:
                if i + 1 < len(words):
                    next_word = words[i + 1]
                    # Try to parse as number word
                    num = self._word_to_number(next_word)
                    if num:
                        return num
                    # Try to parse as digit
                    try:
                        return int(next_word)
                    except ValueError:
                        pass
        
        # Look for standalone numbers that might be dB
        match = re.search(r'(\d+)\s*(?:дб|децибел)', text)
        if match:
            return int(match.group(1))
        
        return None
    
    def _parse_command(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse recognized text into command"""
        text = text.lower().strip()
        logger.info(f"Parsing command: '{text}'")
        
        # Check for volume down commands
        if any(word in text for word in ["тише", "убавь", "убрать"]) or "минус" in text:
            channel = self._extract_channel_number(text)
            db_amount = self._extract_db_amount(text) or 3  # Default 3 dB
            amount = db_amount / 100.0  # Normalize
            
            logger.info(f"✅ Parsed VOLUME_DOWN: channel={channel}, amount={db_amount} dB")
            return {
                "type": "volume_down",
                "channel": channel,
                "amount": amount,
                "db": db_amount
            }
        
        # Check for volume up commands
        if any(word in text for word in ["громче", "прибавь", "добавить"]) or "плюс" in text:
            channel = self._extract_channel_number(text)
            db_amount = self._extract_db_amount(text) or 3  # Default 3 dB
            amount = db_amount / 100.0
            
            logger.info(f"✅ Parsed VOLUME_UP: channel={channel}, amount={db_amount} dB")
            return {
                "type": "volume_up",
                "channel": channel,
                "amount": amount,
                "db": db_amount
            }
        
        # Check for mute commands
        if any(word in text for word in ["мут", "мьют", "заглушить"]):
            channel = self._extract_channel_number(text)
            logger.info(f"✅ Parsed MUTE: channel={channel}")
            return {
                "type": "mute_channel",
                "channel": channel,
                "muted": True
            }
        
        # Check for unmute
        if "включить" in text and any(word in text for word in ["канал", "первый", "второй"]):
            channel = self._extract_channel_number(text)
            logger.info(f"✅ Parsed UNMUTE: channel={channel}")
            return {
                "type": "mute_channel",
                "channel": channel,
                "muted": False
            }
        
        # Check for snapshot load
        if any(word in text for word in ["загрузить", "снапшот", "сцена", "пресет"]):
            # Extract snapshot name (everything after the command word)
            for keyword in ["загрузить", "снапшот", "сцена", "пресет"]:
                if keyword in text:
                    parts = text.split(keyword)
                    if len(parts) > 1:
                        snap_name = parts[1].strip()
                        if snap_name:
                            logger.info(f"✅ Parsed LOAD_SNAP: name={snap_name}")
                            return {
                                "type": "load_snap",
                                "snap_name": snap_name
                            }
        
        # Check for channel selection (fader)
        if "канал" in text or "фейдер" in text:
            channel = self._extract_channel_number(text)
            if channel:
                logger.info(f"✅ Parsed SET_FADER: channel={channel}")
                return {
                    "type": "set_fader",
                    "channel": channel,
                    "value": 0.5  # Default to middle position
                }
        
        # Check for gain
        if "гейн" in text:
            channel = self._extract_channel_number(text)
            if channel:
                db_amount = self._extract_db_amount(text)
                logger.info(f"✅ Parsed SET_GAIN: channel={channel}, db={db_amount}")
                return {
                    "type": "set_gain",
                    "channel": channel,
                    "value": db_amount or 0
                }
        
        logger.warning(f"❌ No command pattern matched for: {text}")
        return None
