import copy
import json
from dataclasses import replace

import pytest
import torch
from torch import nn

from age_gap.common.manifest import file_record
from scripts.mtlface_joint_epoch_v3 import JointEpochFailure, run_joint_epoch
from scripts.run_mtlface_joint_campaign_v3 import CampaignConfig, load_inputs, run_campaign
from tests.test_mtlface_target_stream_v3 import fixture as target_fixture
from tests.test_smoke_mtlface_joint_v3 import Discriminator, Generator, Recognizer


@pytest.fixture
def setup(tmp_path, monkeypatch):
    import scripts.run_mtlface_joint_campaign_v3 as module

    stream = target_fixture(tmp_path)
    paths = {name: tmp_path / f"{name}.json" for name in ("faces", "initialization", "smoke", "joint_smoke", "targets", "weights")}
    for path in paths.values():
        path.write_text("{}", encoding="utf-8")
    faces = dict(
        inputs=[file_record(row.path) for row in stream.records], outputs=[],
        metrics=dict(counts=dict(retained_images=8, retained_people=8)),
    )
    init = dict(inputs=[file_record(paths["weights"])], outputs=[], metrics=dict(identity_classes=8))
    empty = dict(inputs=[], outputs=[])
    contract = dict(weights=file_record(paths["weights"]))
    monkeypatch.setattr(module, "load_inputs", lambda *a: (init, faces, empty, contract, empty, empty))

    def models(weights, classes, *, seed, scope):
        torch.manual_seed(seed)
        return Recognizer(), nn.Linear(3, classes), dict(seed=seed, weights=file_record(weights)), Generator(), Discriminator()

    def dataset(model, path):
        return stream.dataset, faces

    def epoch(model, *args, **kwargs):
        return run_joint_epoch(model, *args, encoder_probe=model.stem, **kwargs)

    config = CampaignConfig(2, 3, 0.001, 0.001, 0.001, "frozen")
    return dict(paths=paths, out=tmp_path / "campaign", config=config,
                model_factory=models, dataset_factory=dataset, epoch_runner=epoch)


def test_three_seed_campaign_binds_all_weights_and_ledgers(setup):
    result = run_campaign(**setup)
    assert result["training_complete"]
    assert result["completed_seeds"] == [42, 1, 2]
    assert not result["full_method_parity"] and not result["common_budget_matched"]
    native = json.loads((setup["out"] / "summary.manifest.json").read_text())
    assert native["parameters"]["implementation_mode"] == "injected-test"
    assert native["parameters"]["checkpoint_selection"] == "last_epoch"
    bound = {r["sha256"] for r in native["outputs"]}
    checkpoints = []
    for seed in result["completed_seeds"]:
        cell = setup["out"] / "private" / f"seed_{seed}"
        checkpoint = cell / "last.pt"
        assert file_record(checkpoint)["sha256"] in bound
        checkpoints.append(torch.load(checkpoint, weights_only=True))
        history = json.loads((cell / "history.json").read_text())
        assert len(history) == 2
        assert history[1]["ledger"]["target_stream_before"] == history[0]["ledger"]["target_stream_after"]
        for epoch in (1, 2):
            path = cell / f"epoch_{epoch}_batches.jsonl"
            assert file_record(path)["sha256"] in bound
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == 3
            assert [r["ledger"]["last_batch_size"] for r in rows] == [3, 3, 2]
            assert rows[-1]["ledger"]["last_batch_summary"]["recognition"]["total"] > 0
    assert set(checkpoints[0]) == {"model", "identity_head", "generator", "discriminator"}
    assert not torch.equal(checkpoints[0]["model"]["stem.weight"], checkpoints[1]["model"]["stem.weight"])


def test_preflight_constructs_no_models(setup):
    def refuse(*args, **kwargs):
        raise AssertionError("preflight must not construct a neural model")

    setup["model_factory"] = refuse
    result = run_campaign(**setup, preflight_only=True)
    assert result["preflight_complete"] and not result["training_complete"]
    assert not (setup["out"] / "private").exists()


def test_failed_epoch_saves_failure_not_completion(setup):
    def fail(*args, **kwargs):
        raise JointEpochFailure("injected partial failure", {"optimizer_steps": {"fr": 1}})

    setup["epoch_runner"] = fail
    with pytest.raises(JointEpochFailure):
        run_campaign(**setup)
    path = setup["out"] / "private" / "seed_42" / "epoch_1_failure.json"
    assert json.loads(path.read_text())["training_complete"] is False
    assert not (path.parent / "cell.manifest.json").exists()
    assert not (setup["out"] / "summary.manifest.json").exists()


def test_wrong_epoch_coverage_refused(setup):
    original = setup["epoch_runner"]

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        result["ledger"]["source_unique_rows"] = 1
        return result

    setup["epoch_runner"] = corrupt
    with pytest.raises(ValueError, match="full joint"):
        run_campaign(**setup)
    assert not (setup["out"] / "summary.manifest.json").exists()


def test_fresh_output_refused(setup):
    setup["out"].mkdir()
    with pytest.raises(FileExistsError):
        run_campaign(**setup)


@pytest.mark.parametrize("key,value", [("epochs", 0), ("epochs", True), ("batch_size", -1), ("fr_lr", float("nan")), ("g_lr", 0), ("seeds", (42, 42)), ("seeds", (-1,)), ("scope", "implicit"), ("generator_bn_policy", "implicit"), ("betas", (1.0, 0.99))])
def test_invalid_config_refused(setup, key, value):
    setup["config"] = replace(setup["config"], **{key: value})
    with pytest.raises(ValueError):
        run_campaign(**setup)
    assert not setup["out"].exists()


def test_partial_batch_logging_refused(setup):
    original = setup["epoch_runner"]

    def skip_progress(*args, **kwargs):
        kwargs["progress"] = None
        return original(*args, **kwargs)

    setup["epoch_runner"] = skip_progress
    with pytest.raises(ValueError, match="persisted batch coverage"):
        run_campaign(**setup)


def test_saved_snapshot_has_independent_storage():
    from scripts.run_mtlface_joint_campaign_v3 import _weights_snapshot

    module = nn.Linear(2, 2)
    snapshot = _weights_snapshot(dict(model=module))
    before = copy.deepcopy(snapshot)
    with torch.no_grad():
        module.weight.add_(1)
    assert torch.equal(snapshot["model"]["weight"], before["model"]["weight"])


@pytest.mark.parametrize("corruption", [None, "joint_incomplete", "target_incomplete", "face_binding", "weight_binding", "target_count", "target_policy", "joint_scope", "training_claim"])
def test_joint_and_target_prerequisite_contracts(setup, monkeypatch, corruption):
    import scripts.run_mtlface_joint_campaign_v3 as module

    paths, config = setup["paths"], setup["config"]
    counts = dict(retained_images=8, retained_people=8)
    faces = dict(metrics=dict(counts=counts))
    init = dict(metrics=dict(identity_classes=8))
    joint = dict(
        inputs=[file_record(paths[name]) for name in ("faces", "initialization", "weights")],
        metrics=dict(real_crop_joint_smoke_complete=True, training_complete=False, scientific_evaluation_complete=False),
        parameters=dict(scope="head", device="cpu"),
    )
    targets = dict(
        inputs=[file_record(paths["faces"])],
        metrics=dict(metadata_sampling_preflight_complete=True, training_complete=False, scientific_evaluation_complete=False,
                     retained_images=8, retained_recorded_identities=8),
        parameters=dict(policy="uniform-group-then-uniform-row-with-replacement"),
    )
    if corruption == "joint_incomplete":
        joint["metrics"]["real_crop_joint_smoke_complete"] = False
    elif corruption == "target_incomplete":
        targets["metrics"]["metadata_sampling_preflight_complete"] = False
    elif corruption == "face_binding":
        targets["inputs"] = []
    elif corruption == "weight_binding":
        # Fixture prerequisite files share bytes, so compare full path+hash records.
        joint["inputs"] = joint["inputs"][:2]
    elif corruption == "target_count":
        targets["metrics"]["retained_images"] = 7
    elif corruption == "target_policy":
        targets["parameters"]["policy"] = "random-unmatched"
    elif corruption == "joint_scope":
        joint["parameters"]["scope"] = "full"
    elif corruption == "training_claim":
        targets["metrics"]["training_complete"] = True
    monkeypatch.setattr(module, "load_prerequisites", lambda **kw: (init, faces, {}, {}))
    monkeypatch.setattr(module, "verified", lambda path, experiment: joint if path == paths["joint_smoke"] else targets)
    if corruption is None:
        assert load_inputs(paths, config)[4:] == (joint, targets)
    else:
        with pytest.raises(ValueError):
            load_inputs(paths, config)


def test_nonfinite_final_snapshot_refused():
    from scripts.run_mtlface_joint_campaign_v3 import _weights_snapshot

    module = nn.Linear(2, 2)
    with torch.no_grad():
        module.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        _weights_snapshot(dict(model=module))
