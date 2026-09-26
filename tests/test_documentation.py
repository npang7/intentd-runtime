"""Check local documentation links and executable Python entry points."""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = [
    ROOT / "README.md",
    ROOT / "docs/design.md",
    ROOT / "tests/README.md",
    ROOT / "bench/README.md",
]
ENTRYPOINTS = [
    "orchestrator.run",
    "orchestrator.resume",
    "bench.chaos",
    "bench.analyze",
    "bench.run_eval",
    "bench.measure_prefix",
    "examples.inspect_journal",
    "examples.recovery_demo",
]


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.name)
def test_local_documentation_links_resolve(document: Path) -> None:
    text = document.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if target.startswith(("https://", "http://", "mailto:", "#")):
            continue
        path = target.split("#", 1)[0]
        assert (document.parent / path).is_file(), f"broken link in {document.name}: {target}"


@pytest.mark.parametrize("module", ENTRYPOINTS)
def test_entrypoint_help_runs_without_credentials(
    module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert callable(importlib.import_module(module).main)
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_documented_focused_test_paths_exist() -> None:
    text = (ROOT / "tests/README.md").read_text(encoding="utf-8")
    for path in re.findall(r"tests/test_[a-z_]+\.py", text):
        assert (ROOT / path).is_file(), f"missing focused check: {path}"
