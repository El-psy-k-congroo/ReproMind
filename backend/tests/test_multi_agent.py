import asyncio
from app.core.model_router import ModelRole, ModelRouter
from app.config import Settings
from app.services.reproduction_engine import extract_repository_urls, validate_repository_url


def test_moma_model_routing_uses_role_specific_models(tmp_path):
    settings = Settings(
        _env_file=None,
        model_routing_path=tmp_path / 'routing.json',
        model_provider="moma",
        moma_api_key="test-key",
        moma_model_chat="moma/chat",
        moma_model_coder="moma/coder",
        moma_model_reasoning="moma/reasoner",
    )
    router = ModelRouter(settings)
    assert router.select(ModelRole.paper).model == "moma/chat"
    assert router.select(ModelRole.code).model == "moma/coder"
    assert router.select(ModelRole.reasoning).model == "moma/reasoner"


def test_moma_accepts_full_chat_completions_url():
    settings = Settings(
        model_provider="moma",
        moma_api_key="test-key",
        moma_base_url="https://example.test/v1/chat/completions",
        moma_model_chat="model-id",
    )
    base_url = ModelRouter(settings).select(ModelRole.paper).base_url
    assert base_url == "https://example.test/v1"


def test_extracts_and_validates_paper_repository_url():
    text = "Code is available at https://github.com/ethanjperez/film."
    assert extract_repository_urls(text) == ["https://github.com/ethanjperez/film"]
    assert validate_repository_url("https://gitee.com/team/project.git") == "https://gitee.com/team/project.git"
