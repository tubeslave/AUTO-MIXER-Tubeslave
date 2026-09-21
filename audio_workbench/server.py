from __future__ import annotations

from typing import Any
from . import core

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise SystemExit("Install MCP Python SDK: pip install 'mcp[cli]'") from exc

mcp = FastMCP("Audio Workbench")

@mcp.tool()
def inspect_audio(project_root: str, audio_path: str) -> dict[str, Any]:
    """Register immutable audio identity and return deterministic signal analysis."""
    identity = core.register(project_root, audio_path)
    return {"identity": identity, "analysis": core.analyze(audio_path)}

@mcp.tool()
def analyze_audio(audio_path: str, include_loudness: bool = True) -> dict[str, Any]:
    """Analyze signal. Metrics are evidence; this tool never claims artistic quality."""
    out = core.analyze(audio_path)
    if include_loudness:
        out["ffmpeg_ebur128"] = core.ffmpeg_loudness(audio_path)
    return out

@mcp.tool()
def record_check(project_root: str, render_sha: str, check_name: str,
                 result: dict[str, Any], status: str = "ok") -> dict[str, Any]:
    """Attach a completed check to one exact render."""
    return core.record_check(project_root, render_sha, check_name, result, status)

@mcp.tool()
def invalidate_after_change(project_root: str, render_sha: str, change_type: str) -> dict[str, Any]:
    """Mark dependent checks stale after a proposed or applied change."""
    return core.invalidate(project_root, render_sha, change_type)

@mcp.tool()
def get_coverage(project_root: str, render_sha: str) -> dict[str, Any]:
    """Return mandatory-check coverage. Finalization is blocked by missing/stale checks."""
    return core.coverage(project_root, render_sha)

@mcp.tool()
def get_next_task(project_root: str, render_sha: str) -> dict[str, Any]:
    """Return the next mandatory verification task."""
    return core.next_task(project_root, render_sha)

@mcp.tool()
def record_decision(project_root: str, render_sha: str, hypothesis: str,
                    action: dict[str, Any], outcome: str = "proposed") -> dict[str, Any]:
    """Persist hypothesis -> action -> outcome for later audit/learning."""
    return core.log_decision(project_root, render_sha, hypothesis, action, outcome)

if __name__ == "__main__":
    mcp.run()
