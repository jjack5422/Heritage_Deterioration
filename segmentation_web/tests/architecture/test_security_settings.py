from pathlib import Path

import pytest

from config import Settings, validate_private_demo_settings


def _settings(**overrides) -> Settings:
    values = {
        "model_root": Path("/data/models"),
        "flask_host": "127.0.0.1",
        "flask_port": 5000,
        "gradio_host": "127.0.0.1",
        "gradio_port": 7860,
        "flask_api_url": "http://127.0.0.1:5000",
        "max_upload_mb": 10,
        "request_timeout_seconds": 120.0,
        "private_demo_mode": True,
        "gradio_auth_username": "meeting",
        "gradio_auth_password": "strong-demo-password",
        "internal_api_key": "c" * 32,
    }
    values.update(overrides)
    return Settings(**values)


def test_private_demo_settings_accept_complete_credentials() -> None:
    settings = _settings()

    validate_private_demo_settings(settings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gradio_auth_username", None),
        ("gradio_auth_password", None),
        ("internal_api_key", None),
        ("internal_api_key", "too-short"),
    ],
)
def test_private_demo_settings_fail_closed(field: str, value: str | None) -> None:
    with pytest.raises(ValueError, match="private demo"):
        validate_private_demo_settings(_settings(**{field: value}))


def test_non_private_mode_does_not_require_credentials() -> None:
    settings = _settings(
        private_demo_mode=False,
        gradio_auth_username=None,
        gradio_auth_password=None,
        internal_api_key=None,
    )

    validate_private_demo_settings(settings)


def test_private_demo_accepts_short_nonempty_password() -> None:
    settings = _settings(gradio_auth_password="1234")

    validate_private_demo_settings(settings)
