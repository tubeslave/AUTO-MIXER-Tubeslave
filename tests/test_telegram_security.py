import importlib.util
from pathlib import Path

import pytest

from tools import telegram_codex_bridge as bridge


def _load_script(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, Path(relative_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "path",
    [
        "plugins/telegram/scripts/get_updates.py",
        "plugins/telegram/scripts/send_message.py",
    ],
)
def test_telegram_plugin_scripts_do_not_accept_token_on_process_argv(path):
    module = _load_script(path.replace("/", "_"), path)

    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--token", "secret-visible-in-ps"])


def test_bridge_authority_is_bound_to_both_chat_and_sender_user():
    update = {
        "message": {
            "chat": {"id": -100123},
            "from": {"id": 42},
            "text": "/newbranch safe",
        }
    }

    assert bridge.extract_message(update) == ("-100123", "42", "/newbranch safe")
    assert bridge.is_authorized_sender(
        "-100123",
        "42",
        allowed_chat_id="-100123",
        allowed_user_id="42",
    ) is True
    assert bridge.is_authorized_sender(
        "-100123",
        "99",
        allowed_chat_id="-100123",
        allowed_user_id="42",
    ) is False
    assert bridge.is_authorized_sender(
        "-100999",
        "42",
        allowed_chat_id="-100123",
        allowed_user_id="42",
    ) is False
