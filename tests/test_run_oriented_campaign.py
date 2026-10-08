import copy

import pytest

from age_gap.common.manifest import file_record
from scripts.run_oriented_campaign import PROTOCOL, training_contract


def example(tmp_path):
    arm = tmp_path / "arm.jsonl"
    arm.write_text("{}\n", encoding="utf-8")
    native = {
        "experiment": "pair-contrastive-backbone-finetune",
        "parameters": PROTOCOL | {"seed": 42, "pairs_file": str(arm)},
        "inputs": [file_record(arm)],
    }
    return arm, native


def test_contract_accepts_exact_protocol(tmp_path):
    arm, native = example(tmp_path)
    training_contract(native, arm, 42)


@pytest.mark.parametrize(
    "key,value",
    [
        ("batchnorm_policy", "adapt_all"),
        ("epochs_executed", 9),
        ("selected_epoch", 8),
        ("checkpoint_selection", "best_val"),
        ("gap_weight", 1.0),
        ("learning_rate", 1e-4),
        ("batch_size", 32),
        ("trainable_scope", "full"),
        ("seed", 1),
    ],
)
def test_reject_protocol_drift(tmp_path, key, value):
    arm, native = example(tmp_path)
    bad = copy.deepcopy(native)
    bad["parameters"][key] = value
    with pytest.raises(ValueError):
        training_contract(bad, arm, 42)


def test_reject_unbound_or_changed_arm(tmp_path):
    arm, native = example(tmp_path)
    native["inputs"] = []
    with pytest.raises(ValueError):
        training_contract(native, arm, 42)
    native["inputs"] = [file_record(arm)]
    arm.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError):
        training_contract(native, arm, 42)
