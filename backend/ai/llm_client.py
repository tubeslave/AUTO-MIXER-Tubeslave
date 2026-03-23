"""
LLM client for AI-assisted mixing decisions.
Supports Ollama (local) and Perplexity (cloud) backends.
"""
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM."""
    text: str
    model: str
    tokens_used: int
    success: bool
    error: Optional[str] = None


class LLMClient:
    """LLM client supporting Ollama and Perplexity backends."""

    def __init__(self,
                 backend: str = 'ollama',
                 model: str = 'llama3',
                 ollama_url: str = 'http://localhost:11434',
                 perplexity_api_key: Optional[str] = None,
                 system_prompt: Optional[str] = None):
        self.backend = backend
        self.model = model
        self.ollama_url = ollama_url.rstrip('/')
        self.perplexity_api_key = perplexity_api_key
        self.system_prompt = system_prompt or self._default_system_prompt()
        self._session = None
        logger.info(f"LLM client initialized: {backend}/{model}")

    def _default_system_prompt(self) -> str:
        return (
            "You are an expert live sound engineer assistant for the AUTO-MIXER system. "
            "You help make mixing decisions for live concerts using a Behringer Wing Rack mixer. "
            "Provide concise, actionable advice based on signal analysis data. "
            "Always prioritize: 1) Safety (no feedback/clipping), 2) Clarity (vocal intelligibility), "
            "3) Balance (appropriate instrument levels), 4) Dynamics (natural feel). "
            "Respond in JSON when asked for parameter recommendations."
        )

    def query(self, prompt: str, context: Optional[str] = None) -> LLMResponse:
        """Send query to LLM backend and return response."""
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"

        if self.backend == 'ollama':
            return self._query_ollama(full_prompt)
        elif self.backend == 'perplexity':
            return self._query_perplexity(full_prompt)
        else:
            return LLMResponse(
                text='', model=self.model, tokens_used=0,
                success=False, error=f"Unknown backend: {self.backend}"
            )

    def _query_ollama(self, prompt: str) -> LLMResponse:
        """Query Ollama local LLM at configured URL."""
        try:
            import requests
        except ImportError:
            return LLMResponse(
                text='', model=self.model, tokens_used=0,
                success=False, error="requests library not installed"
            )

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'system': self.system_prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.3,
                        'num_predict': 512,
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return LLMResponse(
                text=data.get('response', ''),
                model=self.model,
                tokens_used=data.get('eval_count', 0),
                success=True
            )
        except Exception as e:
            logger.error(f"Ollama query error: {e}")
            return LLMResponse(
                text='', model=self.model, tokens_used=0,
                success=False, error=str(e)
            )

    def _query_perplexity(self, prompt: str) -> LLMResponse:
        """Query Perplexity API for cloud-based LLM inference."""
        if not self.perplexity_api_key:
            return LLMResponse(
                text='', model=self.model, tokens_used=0,
                success=False, error="Perplexity API key not set"
            )

        try:
            import requests
        except ImportError:
            return LLMResponse(
                text='', model=self.model, tokens_used=0,
                success=False, error="requests library not installed"
            )

        try:
            response = requests.post(
                'https://api.perplexity.ai/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.perplexity_api_key}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': 'llama-3.1-sonar-small-128k-online',
                    'messages': [
                        {'role': 'system', 'content': self.system_prompt},
                        {'role': 'user', 'content': prompt},
                    ],
                    'max_tokens': 512,
                    'temperature': 0.3,
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            text = data['choices'][0]['message']['content']
            tokens = data.get('usage', {}).get('total_tokens', 0)
            return LLMResponse(
                text=text, model='perplexity', tokens_used=tokens, success=True
            )
        except Exception as e:
            logger.error(f"Perplexity query error: {e}")
            return LLMResponse(
                text='', model=self.model, tokens_used=0,
                success=False, error=str(e)
            )

    def get_mix_recommendation(self, channel_state: Dict, context_entries: Optional[List[str]] = None) -> Dict:
        """Get mixing recommendation for a channel.
        Returns dict with gain_db, eq_bands, comp_threshold, comp_ratio,
        comp_attack_ms, comp_release_ms, pan, reason."""
        context = ""
        if context_entries:
            context = "\n".join(context_entries[:3])

        prompt = (
            f"Given channel state: {json.dumps(channel_state, indent=2)}\n"
            "Recommend mixing parameters as JSON with keys: "
            "gain_db, eq_bands (list of {{freq, gain_db, q}}), "
            "comp_threshold, comp_ratio, comp_attack_ms, comp_release_ms, "
            "pan (-1 to 1), reason."
        )

        response = self.query(prompt, context)
        if response.success:
            try:
                text = response.text
                json_start = text.find('{')
                json_end = text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    return json.loads(text[json_start:json_end])
            except json.JSONDecodeError:
                logger.debug("Could not parse LLM response as JSON, using fallback")

        return self._fallback_recommendation(channel_state)

    def _fallback_recommendation(self, state: Dict) -> Dict:
        """Fallback rule-based recommendation when LLM is unavailable.
        Returns sensible defaults based on instrument type."""
        instrument = state.get('instrument', 'unknown')

        # Instrument-specific defaults based on live sound engineering standards
        defaults = {
            'lead_vocal': {
                'gain_db': -6, 'comp_threshold': -18, 'comp_ratio': 3.0, 'pan': 0.0,
                'eq_bands': [
                    {'freq': 80, 'gain_db': -18, 'q': 0.7},   # HPF
                    {'freq': 250, 'gain_db': -3, 'q': 1.5},    # reduce mud
                    {'freq': 3000, 'gain_db': 2, 'q': 1.0},    # presence
                    {'freq': 8000, 'gain_db': 1.5, 'q': 0.8},  # air
                ],
            },
            'backing_vocal': {
                'gain_db': -10, 'comp_threshold': -16, 'comp_ratio': 3.0, 'pan': 0.0,
                'eq_bands': [
                    {'freq': 100, 'gain_db': -18, 'q': 0.7},
                    {'freq': 250, 'gain_db': -3, 'q': 1.5},
                    {'freq': 3500, 'gain_db': 2, 'q': 1.0},
                ],
            },
            'kick': {
                'gain_db': -8, 'comp_threshold': -16, 'comp_ratio': 4.0, 'pan': 0.0,
                'eq_bands': [
                    {'freq': 60, 'gain_db': 3, 'q': 1.0},     # thump
                    {'freq': 350, 'gain_db': -4, 'q': 1.5},    # boxiness
                    {'freq': 4000, 'gain_db': 3, 'q': 1.2},    # click/attack
                ],
            },
            'snare': {
                'gain_db': -10, 'comp_threshold': -14, 'comp_ratio': 3.0, 'pan': 0.0,
                'eq_bands': [
                    {'freq': 80, 'gain_db': -12, 'q': 0.7},    # HPF
                    {'freq': 200, 'gain_db': 2, 'q': 1.0},     # body
                    {'freq': 900, 'gain_db': -2, 'q': 2.0},    # ring
                    {'freq': 5000, 'gain_db': 3, 'q': 1.0},    # snap
                ],
            },
            'bass_guitar': {
                'gain_db': -8, 'comp_threshold': -16, 'comp_ratio': 3.0, 'pan': 0.0,
                'eq_bands': [
                    {'freq': 40, 'gain_db': -6, 'q': 0.7},     # sub rumble
                    {'freq': 80, 'gain_db': 2, 'q': 1.0},      # fundamental
                    {'freq': 250, 'gain_db': -2, 'q': 1.5},    # mud
                    {'freq': 700, 'gain_db': 2, 'q': 1.0},     # growl
                ],
            },
            'electric_guitar': {
                'gain_db': -12, 'comp_threshold': -14, 'comp_ratio': 2.0, 'pan': -0.3,
                'eq_bands': [
                    {'freq': 80, 'gain_db': -18, 'q': 0.7},    # HPF
                    {'freq': 400, 'gain_db': -2, 'q': 1.5},    # mud
                    {'freq': 2500, 'gain_db': 2, 'q': 1.0},    # presence
                ],
            },
            'acoustic_guitar': {
                'gain_db': -12, 'comp_threshold': -16, 'comp_ratio': 2.0, 'pan': 0.3,
                'eq_bands': [
                    {'freq': 80, 'gain_db': -18, 'q': 0.7},    # HPF
                    {'freq': 200, 'gain_db': -3, 'q': 1.5},    # boominess
                    {'freq': 3000, 'gain_db': 2, 'q': 1.0},    # clarity
                    {'freq': 10000, 'gain_db': 1.5, 'q': 0.8}, # sparkle
                ],
            },
            'keys_piano': {
                'gain_db': -14, 'comp_threshold': -16, 'comp_ratio': 2.0, 'pan': 0.2,
                'eq_bands': [
                    {'freq': 60, 'gain_db': -6, 'q': 0.7},     # rumble
                    {'freq': 250, 'gain_db': -2, 'q': 1.5},    # mud
                    {'freq': 3000, 'gain_db': 1.5, 'q': 1.0},  # presence
                ],
            },
            'hi_hat': {
                'gain_db': -16, 'comp_threshold': -12, 'comp_ratio': 2.0, 'pan': 0.3,
                'eq_bands': [
                    {'freq': 200, 'gain_db': -18, 'q': 0.7},   # HPF
                    {'freq': 6000, 'gain_db': 2, 'q': 1.0},    # shimmer
                ],
            },
            'overhead': {
                'gain_db': -14, 'comp_threshold': -14, 'comp_ratio': 2.0, 'pan': 0.0,
                'eq_bands': [
                    {'freq': 100, 'gain_db': -12, 'q': 0.7},   # HPF
                    {'freq': 400, 'gain_db': -2, 'q': 1.5},    # mud
                    {'freq': 10000, 'gain_db': 2, 'q': 0.8},   # air
                ],
            },
        }

        rec = defaults.get(instrument, {
            'gain_db': -12, 'comp_threshold': -18, 'comp_ratio': 2.0, 'pan': 0.0,
            'eq_bands': [],
        })
        rec['reason'] = 'Fallback defaults (LLM unavailable)'
        rec.setdefault('comp_attack_ms', 10.0)
        rec.setdefault('comp_release_ms', 100.0)
        return rec

    def is_available(self) -> bool:
        """Check if the LLM backend is reachable."""
        if self.backend == 'ollama':
            try:
                import requests
                resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
                return resp.status_code == 200
            except Exception:
                return False
        elif self.backend == 'perplexity':
            return bool(self.perplexity_api_key)
        return False
