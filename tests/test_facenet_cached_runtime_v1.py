from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.facenet_cached_runtime_v1 import cached_runtime


def test_factory_scope_restored_after_failure():
    calls = []

    def factory(name, pretrained):
        calls.append((name, pretrained))
        # The real factory consumes RNG during native initialization too.
        torch.rand(1)
        model = torch.nn.Linear(1, 1)
        model.trainable_scopes = {"head": ("head",)}
        return model

    def dataset(*args, **kwargs):
        return SimpleNamespace(_items=[(Path("a"), Path("b"), 1, 25, 1.0)], identity_groups=["p"])

    trainer = SimpleNamespace(
        make_backbone=factory, ImagePairDataset=dataset, torch_device=lambda: "cuda"
    )
    original = (trainer.make_backbone, trainer.ImagePairDataset, trainer.torch_device)
    features = np.zeros((2, 1792, 1, 1), dtype=np.float32)
    with pytest.raises(RuntimeError, match="stop"):  # noqa: SIM117
        with cached_runtime(trainer, features, {"a": 0, "b": 1}):
            torch.manual_seed(42)
            wrapper = trainer.make_backbone("facenet", pretrained=True)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(42)
            reference = factory("facenet", pretrained=True)
            assert torch.equal(rng, torch.get_rng_state())
            assert torch.equal(wrapper.native.weight, reference.weight)
            assert trainer.torch_device() == "cpu"
            assert trainer.ImagePairDataset(split="train").labels == [1]
            with pytest.raises(ValueError, match="FaceNet"):
                trainer.make_backbone("arcface")
            raise RuntimeError("stop")
    assert (trainer.make_backbone, trainer.ImagePairDataset, trainer.torch_device) == original
