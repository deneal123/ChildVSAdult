"""Native-bound matched-denominator age constants, without altering v1 results."""

import argparse
import json
from pathlib import Path

import numpy as np

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.evaluate_oriented_cuda_v1 import validate_written_inputs
from scripts.person_age_baselines_v2 import constants
from scripts.run_oriented_campaign import paths_from, verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output required")
    natives = []
    for path in args.bindings:
        experiment = json.loads(path.read_text(encoding="utf-8"))["experiment"]
        if experiment not in {"person-disjoint-image-age-diagnostics-v1", "common-image-index-mechanism-v1"}:
            raise ValueError("native age-probe producer required")
        native = verified(path, experiment)
        if native["metrics"].get("execution_complete") is not True:
            raise ValueError("completed native producer required")
        natives.append(native)
    cache = PROJECT_ROOT / "data/external/fgnet_crops.npz"
    inputs = sorted(set([Path(__file__), cache, *args.bindings,
                         PROJECT_ROOT / "scripts/person_age_baselines_v2.py",
                         PROJECT_ROOT / "scripts/run_oriented_campaign.py",
                         PROJECT_ROOT / "scripts/run_restricted_matched_campaign.py",
                         PROJECT_ROOT / "scripts/evaluate_oriented_cuda_v1.py",
                         PROJECT_ROOT / "src/age_gap/common/manifest.py",
                         *(p for m in natives for p in paths_from(m))]))
    before = [file_record(p) for p in inputs]
    with np.load(cache, allow_pickle=False) as data:
        ages, people = data["ages"], data["subjects"].astype(str)
    summaries, private = {}, {}
    for binding, native in zip(args.bindings, natives, strict=True):
        detail = binding.parent / "private/age_probes.json"
        if file_record(cache) not in native["inputs"] or file_record(detail) not in native["outputs"]:
            raise ValueError("metadata and probe output must be bound")
        probes = json.loads(detail.read_text(encoding="utf-8"))
        for group, members in probes.items():
            for name, probe in members.items():
                rows = np.asarray(probe["row_indices"])
                if not np.array_equal(rows, np.arange(len(people))) or probe["persons"] != people.tolist():
                    raise ValueError("full canonical FG-NET person/index binding required")
                key = f"{binding.parent.name}/{group}/{name}"
                if key in summaries:
                    raise ValueError("duplicate probe")
                result = constants(probe, ages)
                private[key] = result
                summaries[key] = dict(probe_mae_image=probe["mae_image"], probe_mae_person=probe["mae_person"],
                                      baselines={k: {m: v[m] for m in ("mae_image", "mae_person")}
                                                 for k, v in result["baselines"].items()})
    if before != [file_record(p) for p in inputs]:
        raise RuntimeError("inputs changed")
    (args.out / "private").mkdir(parents=True)
    detail = args.out / "private/constants.json"
    detail.write_text(json.dumps(private, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    metrics = dict(execution_complete=True, publication_ready=False, probes=summaries,
                   limitation="fixed chronological ages/person folds; no age removal or mechanism conclusion")
    summary = args.out / "summary.json"
    summary.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = args.out / "summary.manifest.json"
    write_experiment_manifest(target, experiment="native-person-age-baselines-v2",
                              parameters=dict(device="cpu", constants="equal-person fit-only mean/median"),
                              metrics=metrics, inputs=inputs, outputs=[summary, detail])
    validate_written_inputs(target, before)
    print(f"completed matched-denominator baselines: {len(summaries)} probes", flush=True)


if __name__ == "__main__":
    main()
