"""Unqualified cached-input head; preserves native standalone checkpoint keys."""

from torch import nn

from scripts.probe_facenet_head_cache_v1 import cached_forward


class CachedFaceNetHead(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native
        self.trainable_scopes = native.trainable_scopes

    @property
    def net(self):
        return self.native.net

    def forward(self, features):
        return cached_forward(self.native, features)

    def state_dict(self, *args, **kwargs):
        return self.native.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.native.load_state_dict(*args, **kwargs)

