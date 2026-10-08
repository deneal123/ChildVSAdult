"""Prescribed-tensor TIP2021 AOFS loss algebra, NOT a CACon reproduction.

Primary: https://wrap.warwick.ac.uk/id/eprint/153968/7/
WRAP-Age-oriented-face-synthesis-conditional-discriminator-pool-adversarial-triplet-2021.pdf
III-D, Eqs4/5 (physical p6), Eqs6/7 and III-E Eqs8/9 (physical p7).
CACon arXiv:2312.11195v2 section2.2 cites this generator; its five-year bins
are not specified here. No encoder, MTFE, generator, discriminator or training
loop exists in this module. Synthetic tests cannot establish full-method parity.

Distances are Euclidean, the second ranking term is NOT hinged. Hard-mining
uses the selected hardest positive/negative also in their cross-distance;
ties use the first prescribed index (declared implementation choice).
Eq7 negative eligibility and context detachment must be supplied explicitly;
the caller must establish different identity and source/target age eligibility.
Losses alone do not route optimizer ownership: D requires fake images/features
detached before its forward; G requires frozen D parameters but a live fake
input graph. A frozen MTFE must still pass gradients through its fake input.
Sum is the triplet paper form; mean is an explicitly requested adaptation.
Literal GAN minimax and non-saturating adaptation are separately named.
"""

import math
from numbers import Real

import torch
from torch.nn import functional as F

INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def _tensor(name, value, ndim=None):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.numel() == 0 or (ndim is not None and value.ndim != ndim)):
        raise ValueError(f"{name}: nonempty floating tensor with required rank")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name}: finite values required")


def _compatible(reference, *values):
    if any(v.dtype != reference.dtype or v.device != reference.device for v in values):
        raise ValueError("matching dtype and device required")


def _nonnegative(name, value):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name}: finite nonnegative real required")


def _reduce(values, reduction):
    _tensor("loss values", values)
    if reduction == "sum":
        result = values.sum()
    elif reduction == "mean":
        result = values.mean()
    else:
        raise ValueError("reduction must be sum (paper) or mean (adaptation)")
    _tensor("reduced loss", result, 0)
    return result


def euclidean(left, right):
    """Cross-bank distances (N,M), with finite zero-distance subgradients."""
    _tensor("left", left, 2)
    _tensor("right", right, 2)
    _compatible(left, right)
    if left.shape[1] != right.shape[1]:
        raise ValueError("matching feature dimensions required")
    result = torch.linalg.vector_norm(left[:, None] - right[None, :], dim=-1)
    _tensor("distances", result, 2)
    return result


def prescribed_triplets(anchor, positive, negative, margin, *, adversarial,
                        reduction="sum"):
    """Eqs4/5 over caller-prescribed aligned (a,p,n) rows; no implicit mining."""
    for name, value in (("anchor", anchor), ("positive", positive), ("negative", negative)):
        _tensor(name, value, 2)
    _compatible(anchor, positive, negative)
    if anchor.shape != positive.shape or anchor.shape != negative.shape:
        raise ValueError("aligned triplet shapes required")
    if type(adversarial) is not bool:
        raise ValueError("explicit boolean adversarial choice required")
    _nonnegative("margin", margin)
    ap = torch.linalg.vector_norm(anchor - positive, dim=-1)
    an = torch.linalg.vector_norm(anchor - negative, dim=-1)
    values = F.relu(margin + ap - an)
    if adversarial:
        values = values + torch.linalg.vector_norm(negative - positive, dim=-1) - an
    return _reduce(values, reduction)


def hard_adversarial_triplet(embeddings, labels, margin, *, reduction="sum"):
    """Eq6; fail if any anchor lacks a positive or negative, never silently drop."""
    distances = euclidean(embeddings, embeddings)
    if (not isinstance(labels, torch.Tensor) or labels.shape != (len(embeddings),)
            or labels.dtype not in INTEGER_DTYPES or labels.device != embeddings.device):
        raise ValueError("one integer label per embedding on the same device required")
    _nonnegative("margin", margin)
    same = labels[:, None] == labels[None, :]
    positives = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    negatives = ~same
    if not positives.any(dim=1).all() or not negatives.any(dim=1).all():
        raise ValueError("every anchor needs eligible positive and negative")
    counts = torch.unique(labels, return_counts=True)[1]
    if not torch.all(counts == counts[0]):
        raise ValueError("paper T-by-S batch requires equal class counts")
    ap, positive_index = distances.masked_fill(~positives, -torch.inf).max(dim=1)
    an, negative_index = distances.masked_fill(~negatives, torch.inf).min(dim=1)
    np_distance = distances[negative_index, positive_index]
    return _reduce(F.relu(margin + ap - an) + np_distance - an, reduction)


def aofs_adversarial_triplet(anchor, synthesized, same_age_bank, target_age_bank,
                             same_age_mask, target_age_mask, margin, *, detach_context,
                             reduction="sum"):
    """Eq7, fixed synthesized positive and hardest eligible union negative.

    Masks (B,Msource)/(B,Mtarget) encode per-anchor age AND different-identity
    eligibility established outside this module; this code cannot verify it.
    Both families must offer a candidate for each anchor. Context detachment
    is a mandatory declared policy, not a recovered author freeze setting.
    """
    for name, value in (("anchor", anchor), ("synthesized", synthesized),
                        ("same_age_bank", same_age_bank), ("target_age_bank", target_age_bank)):
        _tensor(name, value, 2)
    _compatible(anchor, synthesized, same_age_bank, target_age_bank)
    if anchor.shape != synthesized.shape or any(
            v.shape[1] != anchor.shape[1] for v in (same_age_bank, target_age_bank)):
        raise ValueError("compatible feature shapes required")
    if type(detach_context) is not bool:
        raise ValueError("explicit boolean detach_context policy required")
    _nonnegative("margin", margin)
    for bank, mask in ((same_age_bank, same_age_mask), (target_age_bank, target_age_mask)):
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                or mask.shape != (len(anchor), len(bank)) or mask.device != anchor.device
                or not mask.any(dim=1).all()):
            raise ValueError("per-anchor eligible negative required in each family")
    banks = torch.cat((same_age_bank, target_age_bank))
    if detach_context:
        anchor, banks = anchor.detach(), banks.detach()
    allowed = torch.cat((same_age_mask, target_age_mask), dim=1)
    an, indices = euclidean(anchor, banks).masked_fill(~allowed, torch.inf).min(dim=1)
    ap = torch.linalg.vector_norm(anchor - synthesized, dim=-1)
    np_distance = torch.linalg.vector_norm(banks[indices] - synthesized, dim=-1)
    hinge, rank = F.relu(margin + ap - an), np_distance - an
    return dict(loss=_reduce(hinge + rank, reduction), hinge=hinge, rank=rank,
                negative_index=indices, anchor_negative_distance=an)


def discriminator_loss(real_logits, fake_logits):
    """Negative of Eq3/8 (D minimizes); caller owns fake-input detachment."""
    _tensor("real_logits", real_logits)
    _tensor("fake_logits", fake_logits)
    _compatible(real_logits, fake_logits)
    result = _reduce(F.softplus(-real_logits), "mean") + _reduce(F.softplus(fake_logits), "mean")
    _tensor("discriminator loss", result, 0)
    return result


def generator_loss(fake_logits, *, objective="minimax"):
    """Literal Eq3/8 minimax by default; explicitly named non-saturating alternative."""
    _tensor("fake_logits", fake_logits)
    if objective == "minimax":
        return -_reduce(F.softplus(fake_logits), "mean")
    if objective == "non_saturating_adaptation":
        return _reduce(F.softplus(-fake_logits), "mean")
    raise ValueError("objective must be minimax or non_saturating_adaptation")


def selected_cdp_logits(pool_logits, target_labels):
    """Select prescribed (B,K) logits by integer target category; not a CDP model.

    Does not establish architecture, age bins, forward cost or optimizer ownership.
    Gradients reach only selected entries; actual selected-head forwarding is caller work.
    """
    _tensor("pool_logits", pool_logits, 2)
    if (not isinstance(target_labels, torch.Tensor)
            or target_labels.shape != (len(pool_logits),)
            or target_labels.dtype not in INTEGER_DTYPES
            or target_labels.device != pool_logits.device
            or (target_labels < 0).any() or (target_labels >= pool_logits.shape[1]).any()):
        raise ValueError("one valid integer category per row required")
    return pool_logits.gather(1, target_labels.long()[:, None]).squeeze(1)


def overall_generator_loss(image_loss, feature_loss, triplet_loss, *, lambda_feature, lambda_at):
    """Eq9 with explicit coefficients; no implicit author/common-protocol recipe."""
    for name, value in (("image_loss", image_loss), ("feature_loss", feature_loss),
                        ("triplet_loss", triplet_loss)):
        _tensor(name, value, 0)
    _compatible(image_loss, feature_loss, triplet_loss)
    _nonnegative("lambda_feature", lambda_feature)
    _nonnegative("lambda_at", lambda_at)
    result = image_loss + lambda_feature * feature_loss + lambda_at * triplet_loss
    _tensor("overall_loss", result, 0)
    return result
