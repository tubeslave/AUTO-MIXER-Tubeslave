from pathlib import Path

import pytest

from handlers.product_state_handlers import _append_text_beneath_root
from handlers import product_state_handlers


def test_operator_memory_append_stays_beneath_server_root(tmp_path: Path):
    appended = _append_text_beneath_root(
        tmp_path,
        ".ai/memory/project.md",
        "verified memory block",
    )

    target = tmp_path / ".ai/memory/project.md"
    assert appended == len("verified memory block\n".encode("utf-8"))
    assert target.read_text(encoding="utf-8") == (
        "# Project Memory\n\nverified memory block\n"
    )


def test_operator_memory_append_rejects_traversal_and_final_symlink(tmp_path: Path):
    with pytest.raises(ValueError, match="target_path_invalid"):
        _append_text_beneath_root(tmp_path, "../outside.md", "blocked")

    target = tmp_path / ".ai/memory/project.md"
    target.parent.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("preserve", encoding="utf-8")
    target.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink_forbidden"):
        _append_text_beneath_root(
            tmp_path,
            ".ai/memory/project.md",
            "blocked",
        )

    assert outside.read_text(encoding="utf-8") == "preserve"


def test_policy_memory_writer_root_is_code_owned_not_server_or_request_controlled():
    source = Path(product_state_handlers.__file__).read_text(encoding="utf-8")
    writer = source[source.index("def append_sequence_policy_memory_records") :]
    writer = writer[: writer.index("def policy_memory_writer_payload")]

    assert "root_path = _REPO_ROOT" in writer
    assert "_operator_memory_root" not in writer
