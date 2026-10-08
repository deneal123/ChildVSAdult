"""Verify semantic preservation and no neural-library import in presentation tools."""

import ast
import subprocess
import sys
from pathlib import Path

from scripts import common_pair_linkage_light_v1 as light


def function_body(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    body = function.body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # Ignore docstrings, not executable validation.
    return [ast.dump(node, include_attributes=False) for node in body]


def test_pair_validation_semantics_exactly_preserved():
    root = Path(__file__).resolve().parents[1]
    assert function_body(Path(light.__file__), "validate_pairs") == function_body(
        root / "scripts/run_common_mechanism_v1.py", "validate_pairs")


def test_manifest_validation_semantics_exactly_preserved():
    root = Path(__file__).resolve().parents[1]
    assert function_body(Path(light.__file__), "validate_written_inputs") == function_body(
        root / "scripts/evaluate_oriented_cuda_v1.py", "validate_written_inputs")


def test_fresh_presentation_import_does_not_load_torch():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import scripts.render_common_mechanism_v2; "
         "assert 'torch' not in sys.modules; print('no neural import')"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "no neural import"
