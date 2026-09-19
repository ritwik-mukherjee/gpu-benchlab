"""Regression: CLI output must be encodable on a Windows cp1252 console.

`gpu-bench onnx verify` crashed with UnicodeEncodeError on a 'Δ' column header
AFTER saving its evidence, exiting 1 on a passing check. CliRunner captures output
as UTF-8, so the CLI tests could not see it. The same class of bug crashes
torch.onnx.export's verbose output (an emoji). This test guards the whole class:
every string literal in the CLI modules must encode as cp1252.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import gpu_benchlab.cli

CLI_DIR = Path(gpu_benchlab.cli.__file__).parent
MODULES = sorted(CLI_DIR.glob("*.py"))


def test_cli_modules_found() -> None:
    assert {"main.py", "run_cmd.py", "onnx_cmd.py", "models_cmd.py"} <= {m.name for m in MODULES}


@pytest.mark.parametrize("module", MODULES, ids=lambda p: p.name)
def test_every_string_literal_is_cp1252_encodable(module: Path) -> None:
    offenders = []
    for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                node.value.encode("cp1252")
            except UnicodeEncodeError as exc:
                bad = node.value[exc.start : exc.end]
                offenders.append(f"line {node.lineno}: {bad!r} (U+{ord(bad[0]):04X})")
    assert not offenders, f"{module.name} would crash a cp1252 console: {offenders}"


class TestMarkupIsEscaped:
    """Regression: Rich treated '[onnx-export]' / '[torch]' in error text as markup
    and deleted it, so users saw 'install the  extra'. Stored JSON was correct; the
    console lost the text."""

    def _result_with_error(self, message: str):
        from gpu_benchlab.backends.fake import FakeBackend
        from gpu_benchlab.core.config import parse_config
        from gpu_benchlab.core.engine import BenchmarkEngine
        from gpu_benchlab.core.errors import UnavailableError
        from gpu_benchlab.hardware.detect import detect_environment

        class Missing(FakeBackend):
            def validate(self, config, environment):  # type: ignore[no-untyped-def]
                raise UnavailableError(message)

        cfg = parse_config({"name": "x", "backend": "fake", "model": {"name": "m"}})
        engine = BenchmarkEngine(environment=detect_environment(include_frameworks=False))
        return engine.run(Missing(seed=0), cfg)

    def test_error_table_keeps_bracketed_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        from gpu_benchlab.cli import run_cmd

        run_cmd._render_result(self._result_with_error("Install the [onnx-export] extra."))
        out = " ".join(capsys.readouterr().out.split())
        assert "[onnx-export]" in out

    def test_markup_in_messages_is_not_interpreted(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from gpu_benchlab.cli import run_cmd

        run_cmd._render_result(self._result_with_error("literal [red]not a style[/red] text"))
        out = capsys.readouterr().out
        assert "[red]not a style[/red]" in " ".join(out.split())
