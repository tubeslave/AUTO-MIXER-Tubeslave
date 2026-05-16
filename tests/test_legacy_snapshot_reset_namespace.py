from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
LAB_NAMESPACE = BACKEND_ROOT / "lab_only" / "legacy_snapshot_reset"


def test_legacy_snapshot_reset_scripts_are_archived_under_lab_only_namespace():
    archived_names = {
        "load_snap.py",
        "load_snap_v2.py",
        "load_snap_final.py",
        "find_and_load_snap.py",
        "scan_and_load_snap.py",
        "reset_all_channels.py",
        "reset_modules_trim_faders.py",
        "reset_mixer.sh",
    }

    assert LAB_NAMESPACE.is_dir()
    assert archived_names.issubset({path.name for path in LAB_NAMESPACE.iterdir() if path.is_file()})
    assert all(not (BACKEND_ROOT / name).exists() for name in archived_names)
