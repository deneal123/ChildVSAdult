"""Synthetic algebra/gradient tests, not generator/full-method evidence."""

import itertools
import math

import pytest
import torch

from scripts import cacon_tip2021_losses_v1 as loss


def tensor(values, *, grad=False):
    return torch.tensor(values, dtype=torch.float64, requires_grad=grad)


@pytest.mark.parametrize("adversarial", [False, True])
def test_prescribed_triplets_against_scalar_oracle(adversarial):
    a, p, n = tensor([[0.], [1.]]), tensor([[2.], [3.]]), tensor([[3.], [-2.]])
    expected = sum(max(0, .3 + abs(x - y) - abs(x - z))
                   + (abs(z - y) - abs(x - z) if adversarial else 0)
                   for x, y, z in zip([0., 1.], [2., 3.], [3., -2.], strict=True))
    assert loss.prescribed_triplets(a, p, n, .3, adversarial=adversarial).item() == pytest.approx(expected)
    assert loss.prescribed_triplets(a, p, n, .3, adversarial=adversarial,
                                    reduction="mean").item() == pytest.approx(expected / 2)


def test_rank_term_is_unhinged_and_not_anchor_negative_again():
    a, p, n = tensor([[0.]]), tensor([[2.]]), tensor([[3.]])
    assert loss.prescribed_triplets(a, p, n, .3, adversarial=True).item() == -2


def test_hard_mining_matches_bruteforce_cross_distance():
    values, labels = [0., 2., 3., 7., 9., 12.], [0, 0, 1, 1, 2, 2]
    expected = 0.
    for a in range(len(values)):
        p = max((j for j in range(len(values)) if j != a and labels[j] == labels[a]),
                key=lambda j: abs(values[a] - values[j]))
        n = min((j for j in range(len(values)) if labels[j] != labels[a]),
                key=lambda j: abs(values[a] - values[j]))
        an = abs(values[a] - values[n])
        expected += max(0., .3 + abs(values[a] - values[p]) - an) + abs(values[n] - values[p]) - an
    actual = loss.hard_adversarial_triplet(tensor([[v] for v in values]), torch.tensor(labels), .3)
    assert actual.item() == pytest.approx(expected)
    plain = sum(max(0., .3 + abs(values[a] - values[max(
        (j for j in range(6) if j != a and labels[j] == labels[a]),
        key=lambda j: abs(values[a] - values[j]))])
        - min(abs(values[a] - values[j]) for j in range(6) if labels[j] != labels[a])) for a in range(6))
    assert actual.item() != pytest.approx(plain)


@pytest.mark.parametrize("labels", [[0, 0, 0], [0, 1, 1], [0, 1, 2]])
def test_no_silent_invalid_anchor_removal(labels):
    with pytest.raises(ValueError, match="every anchor"):
        loss.hard_adversarial_triplet(tensor([[0.], [1.], [2.]]), torch.tensor(labels), .3)


@pytest.mark.parametrize("adversarial", [False, True])
def test_coincident_triplet_gradients_are_finite(adversarial):
    a, p, n = (tensor([[0., 0.]], grad=True) for _ in range(3))
    loss.prescribed_triplets(a, p, n, .3, adversarial=adversarial).backward()
    assert all(torch.isfinite(v.grad).all() for v in (a, p, n))


def test_coincident_hard_pair_gradients_are_finite():
    x = tensor([[0., 0.]] * 4, grad=True)
    loss.hard_adversarial_triplet(x, torch.tensor([0, 0, 1, 1]), .3).backward()
    assert torch.isfinite(x.grad).all()


def test_distance_gradcheck():
    a, b = tensor([[.4, 1.1], [2.3, -.7]], grad=True), tensor([[1.6, -.1]], grad=True)
    assert torch.autograd.gradcheck(loss.euclidean, (a, b))


@pytest.mark.parametrize("detach", [False, True])
def test_aofs_eligible_mining_and_explicit_gradient_policy(detach):
    a, p, source, target = (tensor(v, grad=True) for v in
        ([[0.], [10.]], [[2.], [12.]], [[0.], [3.], [10.]], [[4.], [14.]]))
    source_mask = torch.tensor([[False, True, False], [False, False, True]])
    target_mask = torch.tensor([[True, False], [False, True]])
    result = loss.aofs_adversarial_triplet(a, p, source, target, source_mask,
                                          target_mask, .3, detach_context=detach)
    assert result["negative_index"].tolist() == [1, 2]
    assert result["loss"].item() == pytest.approx(-2 + 2.3 + 2)
    result["loss"].backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    for v in (a, source, target):
        assert (v.grad is None) == detach
        if v.grad is not None:
            assert torch.isfinite(v.grad).all()


def test_aofs_coincident_gradients_and_first_tie_policy():
    a, p, source, target = (tensor([[0., 0.]], grad=True) for _ in range(4))
    mask = torch.tensor([[True]])
    r = loss.aofs_adversarial_triplet(a, p, source, target, mask, mask, .3, detach_context=False)
    assert r["negative_index"].tolist() == [0]
    r["loss"].backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in (a, p, source, target))


@pytest.mark.parametrize("invalid", ["empty_source", "empty_target", "float_mask", "bad_shape", "no_policy"])
def test_aofs_rejects_underspecified_eligibility(invalid):
    mask1, mask2, detach = torch.tensor([[True]]), torch.tensor([[True]]), True
    if invalid == "empty_source":
        mask1 = torch.tensor([[False]])
    if invalid == "empty_target":
        mask2 = torch.tensor([[False]])
    if invalid == "float_mask":
        mask1 = tensor([[1.]])
    if invalid == "bad_shape":
        mask1 = torch.tensor([True])
    if invalid == "no_policy":
        detach = 1
    with pytest.raises(ValueError):
        loss.aofs_adversarial_triplet(tensor([[0.]]), tensor([[1.]]), tensor([[2.]]),
                                      tensor([[3.]]), mask1, mask2, .3, detach_context=detach)


def test_gan_minimax_and_non_saturating_are_not_conflated():
    real, fake = tensor([-.7, 1.2]), tensor([-.4, .9], grad=True)
    expected = sum(math.log1p(math.exp(-v)) for v in real.tolist()) / 2
    expected += sum(math.log1p(math.exp(v)) for v in fake.tolist()) / 2
    assert loss.discriminator_loss(real, fake).item() == pytest.approx(expected)
    literal = loss.generator_loss(fake)
    assert literal.item() == pytest.approx(-sum(math.log1p(math.exp(v)) for v in fake.tolist()) / 2)
    adapted = loss.generator_loss(fake, objective="non_saturating_adaptation")
    assert adapted.item() == pytest.approx(sum(math.log1p(math.exp(-v)) for v in fake.tolist()) / 2)
    assert literal.item() != adapted.item()
    assert torch.all(torch.autograd.grad(literal, fake)[0] < 0)
    with pytest.raises(ValueError):
        loss.generator_loss(fake, objective="non_saturating")


def test_cdp_gradients_reach_only_selected_entries():
    pool = tensor([[1., 2., 3.], [4., 5., 6.]], grad=True)
    selected = loss.selected_cdp_logits(pool, torch.tensor([2, 0]))
    assert selected.tolist() == [3., 4.]
    selected.sum().backward()
    assert pool.grad.tolist() == [[0., 0., 1.], [1., 0., 0.]]


@pytest.mark.parametrize("labels", [[-1, 0], [3, 0], [[0], [1]]])
def test_cdp_rejects_bad_target_labels(labels):
    with pytest.raises(ValueError):
        loss.selected_cdp_logits(tensor([[1., 2., 3.], [4., 5., 6.]]), torch.tensor(labels))


def test_explicit_overall_coefficients():
    values = tuple(tensor(v, grad=True) for v in [2., 3., -4.])
    r = loss.overall_generator_loss(*values, lambda_feature=1., lambda_at=.001)
    assert r.item() == pytest.approx(4.996)
    r.backward()
    assert [v.grad.item() for v in values] == [1., 1., .001]


@pytest.mark.parametrize("bad", [True, "0.3", -1., float("nan"), float("inf")])
def test_numeric_guards(bad):
    with pytest.raises(ValueError):
        loss.prescribed_triplets(tensor([[0.]]), tensor([[1.]]), tensor([[2.]]), bad,
                                 adversarial=True)
    with pytest.raises(ValueError):
        loss.overall_generator_loss(tensor(1.), tensor(1.), tensor(1.), lambda_feature=bad,
                                     lambda_at=.001)


@pytest.mark.parametrize("bad", [torch.empty(0, 2), torch.empty(2, 0), torch.ones(2, 2, dtype=torch.int64),
                                 tensor([[float("nan")]]), tensor([[float("inf")]])])
def test_distance_input_guards(bad):
    with pytest.raises(ValueError):
        loss.euclidean(bad, tensor([[1., 2.]]))


def test_mismatched_dtype_and_float_identity_rejected():
    with pytest.raises(ValueError, match="dtype"):
        loss.euclidean(tensor([[0.]]), torch.tensor([[1.]], dtype=torch.float32))
    with pytest.raises(ValueError, match="integer"):
        loss.hard_adversarial_triplet(tensor([[0.], [1.], [2.], [3.]]), tensor([0., 0., 1., 1.]), .3)


def test_all_four_prescribed_triplets_can_be_enumerated():
    x, labels = [0., 2., 3., 7.], [0, 0, 1, 1]
    triples = [(a, p, n) for a, p, n in itertools.product(range(4), repeat=3)
               if a != p and labels[a] == labels[p] and labels[a] != labels[n]]
    a, p, n = (tensor([[x[t[i]]] for t in triples]) for i in range(3))
    result = loss.prescribed_triplets(a, p, n, .3, adversarial=True)
    expected = sum(max(0., .3 + abs(x[a] - x[p]) - abs(x[a] - x[n]))
                   + abs(x[n] - x[p]) - abs(x[a] - x[n]) for a, p, n in triples)
    assert result.item() == pytest.approx(expected)


def test_hard_batch_balance_is_not_silently_assumed():
    with pytest.raises(ValueError, match="equal class counts"):
        loss.hard_adversarial_triplet(tensor([[0.], [1.], [2.], [3.], [4.]]),
                                      torch.tensor([0, 0, 1, 1, 1]), .3)


@pytest.mark.parametrize("operation", ["distances", "triplets", "gan", "overall"])
def test_finite_inputs_do_not_permit_overflowed_outputs(operation):
    big = torch.tensor([[torch.finfo(torch.float32).max]], dtype=torch.float32)
    with pytest.raises(ValueError, match="finite"):
        if operation == "distances":
            loss.euclidean(big, -big)
        elif operation == "triplets":
            loss.prescribed_triplets(big, -big, -big, .3, adversarial=True)
        elif operation == "gan":
            loss.discriminator_loss(-big, big)
        else:
            loss.overall_generator_loss(big[0, 0], big[0, 0], big[0, 0],
                                         lambda_feature=1., lambda_at=1.)
