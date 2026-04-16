"""Tests for ai.llm_client module."""
import pytest
import os
import sys
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from ai.llm_client import LLMClient, LLMResponse


class TestLLMClient:
    """Tests for the LLMClient class."""

    def test_init_defaults(self):
        """LLMClient initializes with expected default values."""
        client = LLMClient()
        assert client.backend == 'auto'
        assert client.model == 'gpt-5.4'
        assert client.ollama_url == 'http://localhost:11434'
        assert client.perplexity_api_key is None
        assert client.model_fallbacks[0] == ('kimi_cli', 'default')
        assert ('openai', 'gpt-5.4') in client.model_fallbacks
        assert ('openai', 'gpt-4o-mini') in client.model_fallbacks
        assert ('ollama', 'kimi-k2.5:cloud') in client.model_fallbacks
        assert ('ollama', 'qwen3:30b') in client.model_fallbacks
        assert ('ollama', 'qwen3:0.6b') in client.model_fallbacks
        assert 'live sound engineer' in client.system_prompt.lower()

    def test_init_custom_params(self):
        """LLMClient respects custom initialization parameters."""
        client = LLMClient(
            backend='perplexity',
            model='custom-model',
            ollama_url='http://custom:8080/',
            perplexity_api_key='test-key',
            openai_api_key='openai-key',
            model_fallbacks=['openai:gpt-5.4', 'ollama:kimi-k2.5:cloud'],
            system_prompt='Custom system prompt',
        )
        assert client.backend == 'perplexity'
        assert client.model == 'custom-model'
        assert client.ollama_url == 'http://custom:8080'  # trailing slash stripped
        assert client.perplexity_api_key == 'test-key'
        assert client.openai_api_key == 'openai-key'
        assert client.model_fallbacks == [('openai', 'gpt-5.4'), ('ollama', 'kimi-k2.5:cloud')]
        assert client.system_prompt == 'Custom system prompt'

    def test_query_unknown_backend_returns_error(self):
        """Querying with an unknown backend returns an error response."""
        client = LLMClient(backend='invalid_backend')
        response = client.query("test prompt")
        assert isinstance(response, LLMResponse)
        assert response.success is False
        assert 'Unknown backend' in response.error

    def test_query_ollama_mocked_success(self):
        """Ollama query returns a successful response when requests succeeds."""
        client = LLMClient(backend='ollama')

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Reduce the gain by 3dB on channel 5.',
            'eval_count': 42,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.dict('sys.modules', {'requests': MagicMock()}):
            import requests as mock_requests
            mock_requests.post = MagicMock(return_value=mock_response)
            with patch('ai.llm_client.requests', mock_requests, create=True):
                # Since requests is imported inside the method, we need to mock at import level
                pass

        # Use a more targeted approach: mock the entire method
        with patch.object(client, '_query_ollama') as mock_query:
            mock_query.return_value = LLMResponse(
                text='Reduce the gain by 3dB on channel 5.',
                model='llama3',
                tokens_used=42,
                success=True,
            )
            response = client.query("What should I do with channel 5?")
            assert response.success is True
            assert response.text == 'Reduce the gain by 3dB on channel 5.'
            assert response.tokens_used == 42

    def test_query_ollama_disables_thinking_and_requires_text(self):
        """Ollama thinking models should not be treated as successful with empty text."""
        client = LLMClient(backend='ollama', model='qwen3:0.6b')

        mock_response = MagicMock()
        mock_response.json.return_value = {
            'response': '',
            'thinking': 'internal reasoning only',
            'eval_count': 32,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.dict('sys.modules', {'requests': MagicMock()}):
            import requests as mock_requests
            mock_requests.post = MagicMock(return_value=mock_response)
            response = client.query("Return JSON")

        assert response.success is False
        payload = mock_requests.post.call_args.kwargs['json']
        assert payload['think'] is False

    def test_query_openai_mocked_success(self):
        """OpenAI backend calls the Responses API and extracts output text."""
        client = LLMClient(backend='openai', model='gpt-5.4', openai_api_key='test-key')

        mock_response = MagicMock()
        mock_response.json.return_value = {
            'output_text': '{"gain_db": -8, "reason": "safe correction"}',
            'usage': {'input_tokens': 10, 'output_tokens': 12},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.dict('sys.modules', {'requests': MagicMock()}):
            import requests as mock_requests
            mock_requests.post = MagicMock(return_value=mock_response)
            response = client.query("Recommend correction")

        assert response.success is True
        assert response.model == 'gpt-5.4'
        assert response.tokens_used == 22
        assert 'gain_db' in response.text
        payload = mock_requests.post.call_args.kwargs['json']
        assert payload['model'] == 'gpt-5.4'
        assert payload['reasoning']['effort'] == 'low'

    def test_auto_backend_falls_back_to_ollama_models(self):
        """Auto backend tries GPT first and falls back to configured Ollama models."""
        client = LLMClient(
            backend='auto',
            model='gpt-5.4',
            openai_api_key='test-key',
            model_fallbacks=['openai:gpt-5.4', 'ollama:kimi-k2.5:cloud', 'ollama:qwen3:30b'],
        )

        with patch.object(client, '_query_openai') as mock_openai, \
                patch.object(client, '_query_ollama') as mock_ollama:
            mock_openai.return_value = LLMResponse(
                text='', model='gpt-5.4', tokens_used=0, success=False, error='offline'
            )
            mock_ollama.return_value = LLMResponse(
                text='{"gain_db": -7}', model='kimi-k2.5:cloud', tokens_used=20, success=True
            )

            response = client.query("Recommend correction")

        assert response.success is True
        assert response.model == 'kimi-k2.5:cloud'
        mock_openai.assert_called_once()
        mock_ollama.assert_called_once()
        assert mock_ollama.call_args.kwargs['model'] == 'kimi-k2.5:cloud'

    def test_kimi_cli_backend_uses_subprocess(self):
        """kimi_cli backend invokes Kimi CLI print mode."""
        client = LLMClient(
            backend='kimi_cli',
            kimi_cli_path='/bin/echo',
            kimi_work_dir='/tmp',
        )
        with patch.object(client, '_query_kimi_cli') as mock_kimi:
            mock_kimi.return_value = LLMResponse(
                text='{"gain_db": -3}', model='kimi-cli', tokens_used=0, success=True
            )
            r = client.query("test")
        assert r.success is True
        mock_kimi.assert_called_once()

    def test_query_backend_dispatches_kimi_cli(self):
        """_query_backend routes kimi_cli to _query_kimi_cli."""
        client = LLMClient(backend='auto', kimi_cli_path='/bin/false')
        with patch.object(client, '_query_kimi_cli') as mock_k:
            mock_k.return_value = LLMResponse(text='x', model='m', tokens_used=0, success=True)
            r = client._query_backend('kimi_cli', 'p', 'default')
        assert r.success is True
        mock_k.assert_called_once_with('p', model='default')

    def test_perplexity_without_api_key_returns_error(self):
        """Perplexity backend returns error when no API key is configured."""
        client = LLMClient(backend='perplexity', perplexity_api_key=None)
        response = client.query("test prompt")
        assert response.success is False
        assert 'API key not set' in response.error

    def test_fallback_recommendation_returns_defaults(self):
        """_fallback_recommendation returns sensible defaults for known instruments."""
        client = LLMClient()

        # Lead vocal
        rec = client._fallback_recommendation({'instrument': 'lead_vocal'})
        assert rec['gain_db'] == -6
        assert rec['comp_ratio'] == 3.0
        assert rec['pan'] == 0.0
        assert len(rec['eq_bands']) > 0
        assert rec['reason'] == 'Fallback defaults (LLM unavailable)'
        assert 'comp_attack_ms' in rec
        assert 'comp_release_ms' in rec

        # Unknown instrument gets generic defaults
        rec_unknown = client._fallback_recommendation({'instrument': 'didgeridoo'})
        assert rec_unknown['gain_db'] == -12
        assert rec_unknown['eq_bands'] == []

    def test_get_mix_recommendation_uses_fallback_on_failure(self):
        """get_mix_recommendation falls back to rule-based defaults when LLM fails."""
        client = LLMClient(backend='ollama')

        # Mock _query_ollama to return failure
        with patch.object(client, '_query_ollama') as mock_query:
            mock_query.return_value = LLMResponse(
                text='', model='llama3', tokens_used=0,
                success=False, error='Connection refused',
            )
            state = {'instrument': 'kick'}
            rec = client.get_mix_recommendation(state)
            assert rec['gain_db'] == -8
            assert rec['comp_ratio'] == 4.0
            assert rec['reason'] == 'Fallback defaults (LLM unavailable)'
