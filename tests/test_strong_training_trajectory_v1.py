import pytest

from scripts.strong_training_trajectory_v1 import summarize


def history():
    return [dict(epoch=i, training_loss=1 / i, validation_auc=.95 - .01 * i,
                 mean_gradient_norm=.3, epoch_seconds=1.) for i in range(1, 9)]


def test_last_selection_not_diagnostic_peak_and_no_fabricated_loss():
    result = summarize(history())
    assert result["selected_epoch"] == 8
    assert result["diagnostic_peak_auc_epoch"] == 1
    assert result["validation_loss"] is None
    assert result["training_loss_change"] < 0
    assert result["validation_auc_change"] < 0


@pytest.mark.parametrize("mutation", ["partial", "reordered", "nan", "auc", "negative", "bool"])
def test_invalid_histories_rejected(mutation):
    rows = history()
    if mutation == "partial":
        rows.pop()
    elif mutation == "reordered":
        rows.reverse()
    elif mutation == "nan":
        rows[0]["training_loss"] = float("nan")
    elif mutation == "auc":
        rows[0]["validation_auc"] = 1.1
    elif mutation == "negative":
        rows[0]["mean_gradient_norm"] = -1
    else:
        rows[0]["epoch_seconds"] = True
    with pytest.raises(ValueError):
        summarize(rows)
