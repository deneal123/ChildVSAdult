"""Process-local factory substitution for pending trajectory qualification."""

from contextlib import contextmanager

from scripts.facenet_cached_dataset_v1 import CachedPairDataset
from scripts.facenet_cached_head_v1 import CachedFaceNetHead


@contextmanager
def cached_runtime(trainer, features, crop_rows):
    """Preserve native model initialization RNG and native DataLoader/training code.

    Only for a separate, single-threaded qualification process. Not a production
    cached-training API and never suitable for patching a live campaign process.
    """
    original = (trainer.make_backbone, trainer.ImagePairDataset, trainer.torch_device)

    def backbone(name, pretrained=True):
        if name != "facenet" or pretrained is not True:
            raise ValueError("native pretrained FaceNet required")
        return CachedFaceNetHead(original[0](name, pretrained=pretrained))

    def dataset(*args, **kwargs):
        if kwargs.get("gap_weight", 0.0) != 0.0 or kwargs.get("crops_dir", "faces") != "faces":
            raise ValueError("native unweighted face-crop protocol required")
        # Original metadata constructor consumes no RNG; pixel preprocessing is
        # deterministic. Keep native usable-row filtering and source ordering.
        return CachedPairDataset(original[1](*args, **kwargs), features, crop_rows)

    trainer.make_backbone = backbone
    trainer.ImagePairDataset = dataset
    trainer.torch_device = lambda: "cpu"
    try:
        yield
    finally:
        trainer.make_backbone, trainer.ImagePairDataset, trainer.torch_device = original
