"""CUDA fixed-checkpoint loss, v6: accept declared native-wrapper metadata.

Immutable v5 required identical component/parent parameter key sets, although
the native AdaFace wrapper deliberately adds seven execution fields. V6 checks
every shared component parameter type-exactly, permits only those known parent
extras, and retains every other v5 provenance/output/CUDA/math guard unchanged.
This measures frozen vs last, not historical loss trajectories or a mechanism.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from scripts import measure_checkpoint_validation_cuda_v5 as v5

EXPERIMENT = "fixed-checkpoint-validation-loss-cuda-v6"
NATIVE_WRAPPER_FIELDS = frozenset({
    "device", "threads", "negative", "torch_version", "cuda_version",
    "gpu_name", "cudnn_version",
})


def verify_component_binding_v6(component, checkpoint, params, *, parent_history=None):
    """No training parameter can be missing, mistyped or disagree with parent."""
    v5.v2.verify_component_binding(component, checkpoint, params, parent_history=parent_history)
    cparams = component.get("parameters")
    if type(cparams) is not dict or type(params) is not dict:
        raise ValueError("component and native parameter maps required")
    if set(cparams) - set(params):
        raise ValueError("native omits a component parameter")
    if set(params) - set(cparams) - NATIVE_WRAPPER_FIELDS:
        raise ValueError("undeclared extra native-wrapper parameter")
    for key, value in cparams.items():
        v5._type_exact_equal(value, params[key], "component parameter")
    if parent_history is not None:
        v5._type_exact_equal(component.get("metrics", {}).get("history", []),
                             parent_history, "component history")


def isolated_v5():
    """Fresh v5 globals only; canonical v5/v2 source and module remain unchanged."""
    name = "_checkpoint_validation_v6_isolated_v5"
    spec = importlib.util.spec_from_file_location(name, Path(v5.__file__))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclass annotation lookup requires registration
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    module.EXPERIMENT = EXPERIMENT
    module.MANDATORY_SEEDS = tuple(dict.fromkeys((
        *module.MANDATORY_SEEDS, "scripts.measure_checkpoint_validation_cuda_v6",
    )))
    module.verify_component_binding_v5 = verify_component_binding_v6
    # run() pins its producing wrapper plus the immutable v5 via mandatory AST
    # closure; fallback command metadata names the real v6 entrypoint as well.
    module.__file__ = __file__
    return module


def run(args):
    return isolated_v5().run(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True,
                        help="native training DIRECTORY containing summary.manifest.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(f"dry-run: {EXPERIMENT}; no CUDA probe, no model load, no data preflight")
        return
    try:
        run(args)
    except BaseException as exc:
        raise SystemExit(v5._redact(exc)) from None
    print(f"{EXPERIMENT}: completed")


if __name__ == "__main__":
    main()
