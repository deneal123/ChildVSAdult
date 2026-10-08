"""Guard the inferential scope of measured effects, not just table rounding."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'latex/papers/journal-1-tbiom/en/main.tex'
SUPPLEMENT = ROOT / 'latex/papers/journal-1-tbiom/en/supplement.tex'


def sections():
    main = MAIN.read_text(encoding='utf-8')
    abstract = main.partition(r'\begin{abstract}')[2].partition(r'\end{abstract}')[0]
    conclusion = main.partition(r'\section{Conclusion and Future Work}')[2]
    scope = main.partition(r'\subsection{Scope of the Claims}')[2].partition(r'\section{Limitations}')[0]
    return main, abstract, conclusion, scope


def test_positive_primary_effect_kept_with_per_checkpoint_inference():
    _, abstract, conclusion, scope = sections()
    result = json.loads((ROOT / 'metrics/fgnet_metrics_v2_20261003/summary.json').read_text())
    row = result['large_gap_25plus']['three_checkpoint_aggregate']['roc_auc']
    assert row['delta_ci95'][0] > 0
    assert f"{row['mean']:.4f}" in abstract
    assert f"{row['mean_checkpoint_delta']:+.4f}" in abstract
    assert 'conditional on these fixed checkpoints' in abstract and 'training-seed population' in abstract
    assert 'paired mean-checkpoint subject interval' in conclusion
    assert 'conditional on current weights and protocols' in scope


def test_lfw_auc_support_does_not_become_accuracy_or_low_far_support():
    main, abstract, conclusion, _ = sections()
    result = json.loads((ROOT / 'metrics/lfw_bound_evaluation_20261002/lfw_bound_evaluation.json').read_text())
    aggregate = result['seed_aggregate']
    assert aggregate['roc_auc']['fixed_checkpoint_mean_gain_subject_ci95'][0] > 0
    for metric in ['accuracy_official_folds', 'eer', 'tar@far=0.01', 'tar@far=0.001']:
        lo, hi = aggregate[metric]['fixed_checkpoint_mean_gain_subject_ci95']
        assert lo <= 0 <= hi
    assert 'accuracy and low-FAR gain intervals include zero' in abstract
    assert 'not a reliable accuracy or low-FAR gain' in conclusion
    assert 'every operating point is flat or better' not in main


def test_unmatched_old_control_not_promoted_to_causal_evidence():
    main, abstract, conclusion, _ = sections()
    budget = json.loads((ROOT / 'metrics/matched_arm_image_budget_20261002/summary.json').read_text())
    assert not budget['all_train_image_budget_equal']
    control_path = ROOT / 'metrics/oriented_cuda_fgnet_20261004/summary.json'
    control = json.loads(control_path.read_text())
    binding = json.loads(control_path.with_suffix('.manifest.json').read_text())
    from age_gap.common.manifest import file_record

    assert file_record(control_path) in binding['outputs']
    assert binding['experiment'] == 'oriented-cuda-controls-full-fgnet-roc-v2'
    assert binding['metrics'] == control and control['execution_complete'] is True
    delta = control['large_gap_25plus']['metrics']['roc_auc']['cross_minus_low']
    lo, hi = delta['mean_checkpoint_ci95']
    assert lo < 0 < hi
    assert 'completed full-image-budget within-VK' in abstract
    assert 'inconclusive paired AUC difference' in abstract
    assert 'generic/co-occurrence controls remain absent' in abstract
    assert 'matched-budget retraining remains incomplete' not in abstract
    assert 'does not separate longitudinal supervision' in main
    assert 'not yet for the added value of longitudinal' in conclusion


def test_objective_similarity_not_equivalence_and_fran_not_all_synthesis():
    main, abstract, conclusion, scope = sections()
    for unsupported in ['the gain is invariant', 'objective-robust gains',
                        'synthetic aging cannot replace', 'regardless of source resolution']:
        assert unsupported not in main
    assert 'not demonstrated equivalence' in abstract
    assert 'tested FRAN controls' in scope
    assert 'method-specific synthesis comparison' in conclusion
    assert 'not an equivalence test or a statement about all synthesis methods' in main
    assert 'Real Pairs Outperform Synthetic Aging' not in main
    assert 'coincide under' not in main
    assert 'adds nothing over a plain margin' not in main
    supplement = SUPPLEMENT.read_text(encoding='utf-8')
    assert 'these restricted comparators do not complete full-method reproduction' in supplement
    assert 'Our complete three-seed' not in supplement


def test_crossplatform_paragraph_does_not_claim_uniform_external_improvement():
    main, _, _, _ = sections()
    transfer = main.split(r"\subsection{Cross-Platform Transfer", 1)[1].split(
        r"\begin{table}", 1)[0]
    assert "positive across all three" not in transfer
    assert "AgeDB-30 and CACD-VS operating-point deterioration" in transfer
    assert "transfer is not uniformly positive" in transfer


def test_restricted_sota_does_not_claim_complete_methods_or_equal_compute():
    main, _, _, _ = sections()
    limitations = main.split(r"\textbf{Controlled SOTA scope:}", 1)[1].split(
        r"\end{itemize}", 1)[0]
    assert "not demonstrated equal compute" in limitations
    assert "MTLFace-CP omits joint synthesis" in limitations
    assert "CACon-CP substitutes FRAN" in limitations
    assert "lacks the supervised final-linear stage" in limitations
    assert "not full-method reproductions" in limitations
    supplement = SUPPLEMENT.read_text(encoding='utf-8')
    assert "Our complete three-seed" not in supplement
    assert "MTLFace-CP omits joint synthesis" in supplement
    assert "CACon-CP substitutes" in supplement
    assert "lacks the supervised final-linear stage" in supplement
    assert "Equal compute has not been demonstrated" in supplement


def test_planned_matrix_not_claimed_publicly_preregistered():
    main, _, _, _ = sections()
    supplement = SUPPLEMENT.read_text(encoding='utf-8')
    assert 'preregistered' not in main
    assert 'preregistered' not in supplement
    assert 'planned fixed-budget' in main and 'planned fixed-budget' in supplement


def test_independent_replication_and_sota_limits_are_visible():
    _, _, conclusion, scope = sections()
    assert 'not independent replications of the main FG-NET corpus' in scope
    assert 'not a population of training seeds' in scope
    assert 'pair-level' in scope
    assert 'reimplementations' in conclusion
    assert 'without proving leaderboard superiority' in conclusion
    assert 'blinded identity/annotation audits' in conclusion


def test_compute_claim_does_not_certify_pending_full_sota_training():
    main, _, _, _ = sections()
    compute = main.split(r"\textbf{Compute.}", 1)[1].split(
        r"\textbf{Data availability.}", 1)[0]
    assert "Completed local experiments" in compute
    assert "Full joint MTLFace/CACon training is still pending" in compute
    assert "peak GPU memory and common compute budget have not been validated" in compute
    assert "entire study" not in compute
    assert "full study within" not in compute


def test_lookalike_diagnostics_do_not_become_universal_or_causal_proof():
    main, _, _, _ = sections()
    paragraph = main.split(r"\textbf{The random-impostor protocol", 1)[1].split(
        r"\begin{table}", 1)[0]
    assert "do not establish this ordering for every person" in paragraph
    assert "not a population-wide failure claim" in paragraph
    assert "not miner-independent or causal proof" in paragraph
    assert "incomplete matrix does not establish a general inability" in paragraph
    assert "the difficulty axis belongs to the task, not to the miner" not in paragraph
    assert "At a 25-year gap a person is less similar to themselves" not in paragraph
