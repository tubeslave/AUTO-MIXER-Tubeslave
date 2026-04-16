"""
LLM client for AI-assisted mixing decisions.
Supports automatic fallback across Kimi CLI, OpenAI, Ollama, and Perplexity backends.
"""
import logging
import json
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Any, Tuple
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
    """LLM client with an online-first, local-fallback model chain."""

    def __init__(self,
                 backend: str = 'auto',
                 model: str = 'gpt-5.4',
                 ollama_url: str = 'http://localhost:11434',
                 perplexity_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 openai_url: str = 'https://api.openai.com/v1/responses',
                 model_fallbacks: Optional[List[str]] = None,
                 openai_reasoning_effort: str = 'low',
                 system_prompt: Optional[str] = None,
                 kimi_cli_path: Optional[str] = None,
                 kimi_work_dir: Optional[str] = None,
                 kimi_timeout_sec: float = 120.0):
        self.backend = backend
        self.model = model
        self.ollama_url = ollama_url.rstrip('/')
        self.perplexity_api_key = perplexity_api_key
        self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        self.openai_url = openai_url
        self.openai_reasoning_effort = openai_reasoning_effort
        self.model_fallbacks = self._parse_model_fallbacks(model_fallbacks)
        self.system_prompt = system_prompt or self._default_system_prompt()
        self._session = None
        self.kimi_timeout_sec = float(
            os.environ.get("AUTOMIXER_KIMI_TIMEOUT", str(kimi_timeout_sec))
        )
        raw_kimi = (
            kimi_cli_path
            or os.environ.get("KIMI_CLI_PATH")
            or os.environ.get("AUTOMIXER_KIMI_CLI")
        )
        self.kimi_cli_path = (raw_kimi or shutil.which("kimi") or "").strip()
        self.kimi_work_dir = (
            kimi_work_dir
            or os.environ.get("AUTOMIXER_KIMI_WORK_DIR")
            or ""
        ).strip()
        logger.info(f"LLM client initialized: {backend}/{model}")

    @staticmethod
    def _openai_model_supports_reasoning(model: str) -> bool:
        m = (model or "").lower()
        return m.startswith(("gpt-5", "o1", "o3", "o4"))

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

        if self.backend == 'auto':
            return self._query_auto(full_prompt)
        elif self.backend == 'kimi_cli':
            return self._query_kimi_cli(full_prompt)
        elif self.backend == 'openai':
            return self._query_openai(full_prompt)
        elif self.backend == 'ollama':
            return self._query_ollama(full_prompt)
        elif self.backend == 'perplexity':
            return self._query_perplexity(full_prompt)
        else:
            return LLMResponse(
                text='', model=self.model, tokens_used=0,
                success=False, error=f"Unknown backend: {self.backend}"
            )

    def _parse_model_fallbacks(self, fallbacks: Optional[List[str]]) -> List[Tuple[str, str]]:
        if not fallbacks:
            fallbacks = [
                "kimi_cli:default",
                f"openai:{self.model}",
                "openai:gpt-4o-mini",
                "ollama:kimi-k2.5:cloud",
                "ollama:qwen3:30b",
                "ollama:qwen3:14b",
                "ollama:qwen3:8b",
                "ollama:qwen3:0.6b",
                "ollama:llama3",
            ]

        parsed: List[Tuple[str, str]] = []
        for item in fallbacks:
            if not item:
                continue
            if isinstance(item, dict):
                backend = str(item.get('backend', '')).strip()
                model = str(item.get('model', '')).strip()
            else:
                raw = str(item).strip()
                if ':' in raw:
                    backend, model = raw.split(':', 1)
                else:
                    backend, model = self.backend, raw
            if backend and model:
                parsed.append((backend, model))
        return parsed

    def _query_auto(self, prompt: str) -> LLMResponse:
        """Try preferred online models first, then local Ollama fallbacks."""
        errors = []
        for backend, model in self.model_fallbacks:
            response = self._query_backend(backend, prompt, model)
            if response.success:
                return response
            errors.append(f"{backend}:{model} -> {response.error or 'failed'}")

        return LLMResponse(
            text='',
            model=self.model,
            tokens_used=0,
            success=False,
            error="All LLM fallbacks failed: " + "; ".join(errors[-5:]),
        )

    def _query_backend(self, backend: str, prompt: str, model: str) -> LLMResponse:
        if backend == 'kimi_cli':
            return self._query_kimi_cli(prompt, model=model)
        if backend == 'openai':
            return self._query_openai(prompt, model=model)
        if backend == 'ollama':
            return self._query_ollama(prompt, model=model)
        if backend == 'perplexity':
            return self._query_perplexity(prompt, model=model)
        return LLMResponse(
            text='',
            model=model,
            tokens_used=0,
            success=False,
            error=f"Unknown fallback backend: {backend}",
        )

    def _resolve_kimi_work_dir(self) -> str:
        """Dedicated work directory so Kimi CLI does not scan the project cwd."""
        if self.kimi_work_dir:
            return os.path.abspath(self.kimi_work_dir)
        here = os.path.dirname(os.path.abspath(__file__))
        repo = os.path.abspath(os.path.join(here, "..", ".."))
        return os.path.join(repo, ".automixer", "kimi_work")

    @staticmethod
    def _strip_kimi_footer(text: str) -> str:
        lines = []
        for line in (text or "").splitlines():
            if line.strip().startswith("To resume this session:"):
                break
            lines.append(line)
        return "\n".join(lines).strip()

    def _query_kimi_cli(self, prompt: str, model: Optional[str] = None) -> LLMResponse:
        """Run Moonshot Kimi CLI in non-interactive print mode (see `kimi --print`)."""
        if not self.kimi_cli_path:
            return LLMResponse(
                text="",
                model="kimi-cli",
                tokens_used=0,
                success=False,
                error="kimi CLI not found (install or set KIMI_CLI_PATH)",
            )
        work = self._resolve_kimi_work_dir()
        try:
            os.makedirs(work, exist_ok=True)
        except OSError as exc:
            return LLMResponse(
                text="",
                model="kimi-cli",
                tokens_used=0,
                success=False,
                error=f"kimi work dir: {exc}",
            )
        user_text = f"{self.system_prompt}\n\n{prompt}" if self.system_prompt else prompt
        cmd: List[str] = [
            self.kimi_cli_path,
            "-w",
            work,
            "--print",
            "--output-format",
            "text",
            "--final-message-only",
            "-p",
            user_text,
        ]
        m = (model or "default").strip()
        if m and m.lower() not in ("default", "auto"):
            cmd.insert(1, "-m")
            cmd.insert(2, m)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max(5.0, self.kimi_timeout_sec),
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            return LLMResponse(
                text="",
                model=m or "kimi-cli",
                tokens_used=0,
                success=False,
                error="kimi CLI timeout",
            )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "kimi CLI failed").strip()
            return LLMResponse(
                text="",
                model=m or "kimi-cli",
                tokens_used=0,
                success=False,
                error=err[:800],
            )
        out = self._strip_kimi_footer((proc.stdout or "").strip())
        if not out and (proc.stderr or "").strip():
            err = (proc.stderr or "").strip()
            return LLMResponse(
                text="",
                model=m or "kimi-cli",
                tokens_used=0,
                success=False,
                error=err[:800],
            )
        return LLMResponse(
            text=out,
            model=m or "kimi-cli",
            tokens_used=0,
            success=bool(out),
        )

    def _kimi_cli_available(self) -> bool:
        path = self.kimi_cli_path
        return bool(path) and os.path.isfile(path)

    def _query_openai(self, prompt: str, model: Optional[str] = None) -> LLMResponse:
        """Query OpenAI Responses API."""
        model = model or self.model
        if not self.openai_api_key:
            return LLMResponse(
                text='', model=model, tokens_used=0,
                success=False, error="OpenAI API key not set"
            )

        try:
            import requests
        except ImportError:
            return LLMResponse(
                text='', model=model, tokens_used=0,
                success=False, error="requests library not installed"
            )

        payload = {
            'model': model,
            'input': [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': prompt},
            ],
            'max_output_tokens': 700,
        }
        # Responses API `reasoning` is only valid for reasoning-capable models.
        if self.openai_reasoning_effort and self._openai_model_supports_reasoning(model):
            payload['reasoning'] = {'effort': self.openai_reasoning_effort}

        try:
            response = requests.post(
                self.openai_url,
                headers={
                    'Authorization': f'Bearer {self.openai_api_key}',
                    'Content-Type': 'application/json',
                },
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            text = self._extract_openai_text(data)
            usage = data.get('usage', {}) or {}
            tokens = (
                usage.get('total_tokens')
                or usage.get('total_tokens_used')
                or usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
            )
            return LLMResponse(text=text, model=model, tokens_used=tokens, success=bool(text))
        except Exception as e:
            logger.error(f"OpenAI query error ({model}): {e}")
            return LLMResponse(text='', model=model, tokens_used=0, success=False, error=str(e))

    def _extract_openai_text(self, data: Dict[str, Any]) -> str:
        if data.get('output_text'):
            return data['output_text']

        chunks = []
        for item in data.get('output', []) or []:
            for content in item.get('content', []) or []:
                if isinstance(content, dict):
                    text = content.get('text') or content.get('content')
                    if text:
                        chunks.append(text)
        return "\n".join(chunks)

    def _query_ollama(self, prompt: str, model: Optional[str] = None) -> LLMResponse:
        """Query Ollama local LLM at configured URL."""
        model = model or self.model
        try:
            import requests
        except ImportError:
            return LLMResponse(
                text='', model=model, tokens_used=0,
                success=False, error="requests library not installed"
            )

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    'model': model,
                    'prompt': prompt,
                    'system': self.system_prompt,
                    'stream': False,
                    'think': False,
                    'options': {
                        'temperature': 0.3,
                        'num_predict': 512,
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            text = data.get('response', '')
            return LLMResponse(
                text=text,
                model=model,
                tokens_used=data.get('eval_count', 0),
                success=bool(text)
            )
        except Exception as e:
            logger.error(f"Ollama query error ({model}): {e}")
            return LLMResponse(
                text='', model=model, tokens_used=0,
                success=False, error=str(e)
            )

    def _query_perplexity(self, prompt: str, model: Optional[str] = None) -> LLMResponse:
        """Query Perplexity API for cloud-based LLM inference."""
        model = model or 'llama-3.1-sonar-small-128k-online'
        if not self.perplexity_api_key:
            return LLMResponse(
                text='', model=model, tokens_used=0,
                success=False, error="Perplexity API key not set"
            )

        try:
            import requests
        except ImportError:
            return LLMResponse(
                text='', model=model, tokens_used=0,
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
                    'model': model,
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
                text=text, model=model, tokens_used=tokens, success=True
            )
        except Exception as e:
            logger.error(f"Perplexity query error ({model}): {e}")
            return LLMResponse(
                text='', model=model, tokens_used=0,
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
            "Recommend small, reversible live-sound corrections as JSON only. "
            "Do not invent missing meter data. Use only keys that should change: "
            "gain_db, eq_bands (list of {{freq, gain_db, q}}), "
            "comp_threshold, comp_ratio, comp_attack_ms, comp_release_ms, "
            "pan (-1 to 1), reason, expected_effect, rollback_hint, risk "
            "(low, medium, or high)."
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
        rec['expected_effect'] = 'Conservative baseline starting point for the channel'
        rec['rollback_hint'] = 'Return fader, EQ, compressor, and pan to the previous stored values'
        rec['risk'] = 'medium'
        rec['llm_available'] = False
        rec.setdefault('comp_attack_ms', 10.0)
        rec.setdefault('comp_release_ms', 100.0)
        return rec

    def is_available(self) -> bool:
        """Check if the LLM backend is reachable."""
        if self.backend == 'auto':
            return any(
                (backend == 'kimi_cli' and self._kimi_cli_available())
                or (backend == 'openai' and bool(self.openai_api_key))
                or (backend == 'ollama' and self._ollama_available())
                or (backend == 'perplexity' and bool(self.perplexity_api_key))
                for backend, _model in self.model_fallbacks
            )
        if self.backend == 'kimi_cli':
            return self._kimi_cli_available()
        if self.backend == 'openai':
            return bool(self.openai_api_key)
        if self.backend == 'ollama':
            return self._ollama_available()
        elif self.backend == 'perplexity':
            return bool(self.perplexity_api_key)
        return False

    def _ollama_available(self) -> bool:
        try:
            import requests
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
