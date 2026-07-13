# -*- mode: python ; coding: utf-8 -*-

import importlib.util
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


backend_root = Path(SPECPATH).resolve()

runtime_identity_spec = importlib.util.spec_from_file_location(
    "automixer_build_runtime_identity",
    backend_root / "runtime_identity.py",
)
if runtime_identity_spec is None or runtime_identity_spec.loader is None:
    raise RuntimeError("Unable to load runtime identity builder")
runtime_identity = importlib.util.module_from_spec(runtime_identity_spec)
runtime_identity_spec.loader.exec_module(runtime_identity)
runtime_source_components = runtime_identity.capture_source_runtime_components(
    backend_root.parent
)
runtime_source_manifest = runtime_identity.build_packaged_runtime_source_manifest(
    runtime_source_components
)
runtime_manifest_root = Path(
    os.environ.get("AUTOMIXER_BACKEND_WORK_DIR", "build")
)
if not runtime_manifest_root.is_absolute():
    runtime_manifest_root = backend_root / runtime_manifest_root
runtime_manifest_root.mkdir(parents=True, exist_ok=True)
runtime_manifest_path = (
    runtime_manifest_root
    / runtime_identity.PACKAGED_RUNTIME_SOURCE_MANIFEST_FILENAME
)
runtime_manifest_path.write_bytes(
    runtime_identity.canonical_json_bytes(runtime_source_manifest) + b"\n"
)

hiddenimports = []
for package in (
    "ai",
    "auto_fader_v2",
    "evaluation",
    "handlers",
    "ml",
    "osc",
    "perceptual",
    "source_knowledge",
):
    hiddenimports.extend(collect_submodules(package))

a = Analysis(
    ["server.py"],
    pathex=[str(backend_root)],
    binaries=[],
    datas=[(str(runtime_manifest_path), ".")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib.tests",
        "numpy.tests",
        "pytest",
        "scipy.tests",
        "torch.testing",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="server",
)
