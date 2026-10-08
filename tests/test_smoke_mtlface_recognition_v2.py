import pytest
import torch
from torch.utils.data import TensorDataset

from scripts.mtlface_training_v2 import FaceRecord
from scripts.smoke_mtlface_recognition_v2 import smoke_step
from tests.test_mtlface_training_v2 import joint_model_and_batch


def dataset_with_metadata(batch):
    dataset = TensorDataset(*batch)
    dataset.records = [
        FaceRecord(None, int(identity), None if int(age) == -1 else int(age))
        for identity, age in zip(batch[1], batch[2], strict=True)
    ]
    return dataset


def test_smoke_checks_updates_buffers_and_selection():
    model, head, batch = joint_model_and_batch()
    metrics, indices = smoke_step(model, head, dataset_with_metadata(batch))
    assert indices == [0, 3]
    assert metrics["frozen_buffers_unchanged"] and metrics["embeddings_finite"]
    assert all(metrics["updated_parameter_tensors"].values())


def test_smoke_refuses_no_missing_age():
    model, head, batch = joint_model_and_batch()
    batch = batch[:2] + (torch.zeros_like(batch[2]),)
    with pytest.raises(ValueError, match="both explicit and missing"):
        smoke_step(model, head, dataset_with_metadata(batch))
