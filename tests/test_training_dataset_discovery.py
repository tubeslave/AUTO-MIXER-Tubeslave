from scripts.discover_training_datasets import (
    build_hf_file_url,
    safe_dataset_dir,
    select_supported_files,
)


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
