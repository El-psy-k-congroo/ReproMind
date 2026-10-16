import asyncio
from types import SimpleNamespace
import pytest
from fastapi import HTTPException
from app.api import models_catalog as catalog


def test_configured_model_does_not_block_on_cold_catalog(monkeypatch):
    monkeypatch.setattr(catalog, 'get_settings', lambda: SimpleNamespace(model_provider='moma', moma_model_chat='configured'))
    async def unavailable(): pytest.fail('Configured model should not contact catalog')
    monkeypatch.setattr(catalog, 'available_models', unavailable)
    asyncio.run(catalog.validate_model('configured'))


def test_directory_timeout_is_not_proof_model_was_removed(monkeypatch):
    monkeypatch.setattr(catalog, '_catalog', None)
    monkeypatch.setattr(catalog, 'get_settings', lambda: SimpleNamespace(model_provider='moma', moma_api_key='offline'))
    async def fallback_only(): return ['default']
    monkeypatch.setattr(catalog, 'available_models', fallback_only)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(catalog.validate_model('other'))
    assert exc.value.status_code == 503
