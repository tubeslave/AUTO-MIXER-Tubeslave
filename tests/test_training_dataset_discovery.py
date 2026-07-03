from scripts.discover_training_datasets import (
    build_hf_file_url,
    safe_dataset_dir,
    safe_dataset_file_path,
    select_supported_files,
)

import pytest


def test_safe_dataset_dir_removes_path_separators():
    assert safe_dataset_dir("mclemcrew/MixAssist") == "mclemcrew__MixAssist"


def test_hf_file_url_quotes_spaces_but_keeps_subdirectories():
    url = build_hf_file_url("mclemcrew/MixParams", "data/train file.parquet")
    assert url == (
        "https://huggingface.co/datasets/mclemcrew/MixParams/resolve/main/"
        "data/train%20file.parquet?download=true"
    )


def test_select_supported_files_skips_hidden_and_keeps_data_files():
    siblings = [
        {"rfilename": ".gitattributes"},
        {"rfilename": "README.md"},
        {"rfilename": "data/train-00000-of-00001.parquet"},
        {"rfilename": "audio/raw.wav"},
        {"rfilename": "metadata.json"},
    ]

    assert select_supported_files(siblings) == [
        "README.md",
        "data/train-00000-of-00001.parquet",
        "metadata.json",
    ]


def test_safe_dataset_file_path_keeps_downloads_inside_dataset_dir(tmp_path):
    dataset_dir = tmp_path / "dataset"

    assert safe_dataset_file_path(dataset_dir, "data/train file.parquet") == (
        dataset_dir / "data" / "train file.parquet"
    )


@pytest.mark.parametrize(
    "filename",
    [
        "../evil.jsonl",
        "data/../../evil.jsonl",
        "/tmp/evil.jsonl",
        "C:/tmp/evil.jsonl",
        r"..\\evil.jsonl",
        "data//evil.jsonl",
    ],
)
def test_safe_dataset_file_path_rejects_unsafe_names(tmp_path, filename):
    with pytest.raises(ValueError, match="unsafe dataset filename"):
        safe_dataset_file_path(tmp_path / "dataset", filename)

    with pytest.raises(ValueError, match="unsafe dataset filename"):
        build_hf_file_url("mclemcrew/MixParams", filename)


def test_select_supported_files_skips_unsafe_names():
    siblings = [
        {"rfilename": "data/train.parquet"},
        {"rfilename": "../evil.jsonl"},
        {"rfilename": "/tmp/evil.csv"},
        {"rfilename": r"..\\evil.json"},
    ]

    assert select_supported_files(siblings) == ["data/train.parquet"]
