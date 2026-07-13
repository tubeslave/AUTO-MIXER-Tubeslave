#!/bin/bash

set -e

TARGET_ARCH="${AUTOMIXER_TARGET_ARCH:-}"
PYTHON_BIN="${AUTOMIXER_PYTHON_BIN:-python3}"
VENV_DIR="${AUTOMIXER_BACKEND_VENV_DIR:-venv}"
REQUIREMENTS_FILE="${AUTOMIXER_BACKEND_REQUIREMENTS_FILE:-requirements.txt}"
BUILD_REQUIREMENTS_FILE="${AUTOMIXER_BACKEND_BUILD_REQUIREMENTS_FILE:-}"
PYINSTALLER_VERSION="${AUTOMIXER_PYINSTALLER_VERSION:-6.20.0}"
PIP_VERSION="${AUTOMIXER_PIP_VERSION:-}"
REQUIRED_PYTHON_VERSION="${AUTOMIXER_REQUIRED_PYTHON_VERSION:-}"
VERIFY_LOCKED_ENV="${AUTOMIXER_VERIFY_LOCKED_ENV:-false}"
WORK_DIR="${AUTOMIXER_BACKEND_WORK_DIR:-build}"
DIST_DIR="${AUTOMIXER_BACKEND_DIST_DIR:-dist}"

normalize_arch() {
    case "$1" in
        arm64|aarch64) echo "arm64" ;;
        x64|x86_64|amd64) echo "x64" ;;
        *) echo "$1" ;;
    esac
}

if [ -n "$TARGET_ARCH" ]; then
    TARGET_ARCH="$(normalize_arch "$TARGET_ARCH")"
    HOST_ARCH="$(normalize_arch "$(uname -m)")"
    if [ "$TARGET_ARCH" != "arm64" ] && [ "$TARGET_ARCH" != "x64" ]; then
        echo "Unsupported AUTOMIXER_TARGET_ARCH: $TARGET_ARCH"
        exit 1
    fi
    if [ "$HOST_ARCH" != "$TARGET_ARCH" ]; then
        echo "Refusing cross-architecture backend build: host=$HOST_ARCH target=$TARGET_ARCH"
        echo "Build the backend on a native target runner or provide a verified universal binary."
        exit 1
    fi
    if [ ! -x /usr/bin/lipo ]; then
        echo "Cannot verify packaged backend architecture: /usr/bin/lipo is unavailable"
        exit 1
    fi
fi

echo "Building Python backend with PyInstaller..."

if [ -n "$REQUIRED_PYTHON_VERSION" ]; then
    ACTUAL_PYTHON_VERSION="$("$PYTHON_BIN" -c 'import platform; print(platform.python_version())')"
    if [ "$ACTUAL_PYTHON_VERSION" != "$REQUIRED_PYTHON_VERSION" ]; then
        echo "Python version mismatch: required=$REQUIRED_PYTHON_VERSION actual=$ACTUAL_PYTHON_VERSION"
        exit 1
    fi
fi

cd backend

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Requirements file not found: $REQUIREMENTS_FILE"
    deactivate
    exit 1
fi
if [ -n "$PIP_VERSION" ]; then
    ACTUAL_PIP_VERSION="$(python -m pip --version | awk '{print $2}')"
    if [ "$ACTUAL_PIP_VERSION" != "$PIP_VERSION" ]; then
        echo "pip version mismatch: required=$PIP_VERSION actual=$ACTUAL_PIP_VERSION"
        deactivate
        exit 1
    fi
fi
if [ "$VERIFY_LOCKED_ENV" = "true" ]; then
    if [ -z "$BUILD_REQUIREMENTS_FILE" ] || [ ! -f "$BUILD_REQUIREMENTS_FILE" ]; then
        echo "Hashed build requirements file not found: $BUILD_REQUIREMENTS_FILE"
        deactivate
        exit 1
    fi
    python -m pip --isolated --disable-pip-version-check install \
        --no-cache-dir --require-hashes -r "$BUILD_REQUIREMENTS_FILE"
    python -m pip --isolated --disable-pip-version-check install \
        --no-cache-dir --require-hashes --no-build-isolation -r "$REQUIREMENTS_FILE"
else
    python -m pip install -r "$REQUIREMENTS_FILE"
    python -m pip install "pyinstaller==$PYINSTALLER_VERSION"
fi
python -m pip --version
ACTUAL_PYINSTALLER_VERSION="$(python -m PyInstaller --version)"
if [ "$ACTUAL_PYINSTALLER_VERSION" != "$PYINSTALLER_VERSION" ]; then
    echo "PyInstaller version mismatch: required=$PYINSTALLER_VERSION actual=$ACTUAL_PYINSTALLER_VERSION"
    deactivate
    exit 1
fi
python -m pip check

if [ "$VERIFY_LOCKED_ENV" = "true" ]; then
    python - "$REQUIREMENTS_FILE" <<'PY'
import importlib.metadata
import pathlib
import re
import shlex
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

lock_path = pathlib.Path(sys.argv[1])
allowed_options = {"--only-binary=:all:", "--no-binary=watchdog"}
logical_lines = []
pending = ""
for raw in lock_path.read_text(encoding="utf-8").splitlines():
    stripped = raw.strip()
    if not stripped or stripped.startswith("#"):
        continue
    pending = f"{pending} {stripped}".strip()
    if pending.endswith("\\"):
        pending = pending[:-1].rstrip()
        continue
    logical_lines.append(pending)
    pending = ""
if pending:
    raise SystemExit("Hashed lock ends with an incomplete continuation")

expected = {}
seen_options = set()
for line in logical_lines:
    if line.startswith("--"):
        if line not in allowed_options:
            raise SystemExit(f"Unsupported hashed-lock option: {line}")
        seen_options.add(line)
        continue
    tokens = shlex.split(line)
    requirement = Requirement(tokens[0])
    hashes = tokens[1:]
    if (
        requirement.url
        or requirement.marker
        or requirement.extras
        or len(requirement.specifier) != 1
        or next(iter(requirement.specifier)).operator != "=="
        or "*" in next(iter(requirement.specifier)).version
        or not hashes
        or any(not re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", value) for value in hashes)
    ):
        raise SystemExit(f"Invalid exact hashed requirement: {line}")
    key = canonicalize_name(requirement.name)
    version = next(iter(requirement.specifier)).version
    if key in expected:
        raise SystemExit(f"Duplicate hashed requirement: {key}")
    expected[key] = version
if seen_options != allowed_options:
    raise SystemExit(f"Hashed-lock options mismatch: {sorted(seen_options)}")

actual = {}
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name")
    if not name:
        continue
    key = canonicalize_name(name)
    if key == "pip":
        continue
    actual[key] = distribution.version
if actual != expected:
    missing = sorted(set(expected.items()) - set(actual.items()))
    unexpected = sorted(set(actual.items()) - set(expected.items()))
    raise SystemExit(
        "Locked environment mismatch; "
        f"missing={missing[:10]} unexpected={unexpected[:10]}"
    )
print(f"Locked environment verified: {len(actual)} distributions")
PY
fi

echo "Building executable with PyInstaller..."
python -m PyInstaller --clean --workpath "$WORK_DIR" --distpath "$DIST_DIR" server.spec

BACKEND_BUNDLE="$DIST_DIR/server"
BACKEND_BINARY="$BACKEND_BUNDLE/server"
RUNTIME_SOURCE_MANIFEST="$BACKEND_BUNDLE/_internal/automixer-runtime-source-manifest.json"
if [ -f "$BACKEND_BINARY" ]; then
    test -d "$BACKEND_BUNDLE/_internal" || {
        echo "Backend onedir bundle is missing _internal runtime"
        deactivate
        exit 1
    }
    test -f "$RUNTIME_SOURCE_MANIFEST" || {
        echo "Backend runtime source manifest is missing: $RUNTIME_SOURCE_MANIFEST"
        deactivate
        exit 1
    }
    echo "✓ Backend onedir bundle built successfully: $BACKEND_BUNDLE"
    echo "✓ Backend executable: $BACKEND_BINARY"
    echo "✓ Backend runtime source manifest: $RUNTIME_SOURCE_MANIFEST"
else
    echo "✗ Build failed!"
    deactivate
    exit 1
fi

if [ -n "$TARGET_ARCH" ]; then
    BACKEND_ARCHS="$(/usr/bin/lipo -archs "$BACKEND_BINARY")"
    REQUIRED_LIPO_ARCH="$TARGET_ARCH"
    if [ "$TARGET_ARCH" = "x64" ]; then
        REQUIRED_LIPO_ARCH="x86_64"
    fi
    case " $BACKEND_ARCHS " in
        *" $REQUIRED_LIPO_ARCH "*) ;;
        *)
            echo "Backend architecture mismatch: target=$TARGET_ARCH binary=$BACKEND_ARCHS"
            deactivate
            exit 1
            ;;
    esac
    echo "✓ Backend architecture verified: $BACKEND_ARCHS"
fi

deactivate
cd ..

echo "Backend build complete!"
