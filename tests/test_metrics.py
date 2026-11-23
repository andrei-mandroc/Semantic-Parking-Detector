import pytest

from src.evaluation.metrics import evaluate_occupancy, summarize_evaluations


def test_confusion_counts_and_rates():
    ground_truth = {"A": True, "B": False, "C": True, "D": False}
    predicted = {"A": True, "B": False, "C": False, "D": True}

    result = evaluate_occupancy(predicted, ground_truth)

    assert {name: result[name] for name in ("tp", "tn", "fp", "fn")} == {
        "tp": 1, "tn": 1, "fp": 1, "fn": 1
    }
    assert result["evaluated_spots"] == 4
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def test_missing_and_unlabeled_ids_are_reported_and_excluded():
    result = evaluate_occupancy(
        {"A": True, "C": False}, {"A": True, "B": False}
    )

    assert result["evaluated_spots"] == 1
    assert result["tp"] == 1
    assert result["missing_prediction_ids"] == ["B"]
    assert result["prediction_only_ids"] == ["C"]
    assert result["accuracy"] == 1.0


def test_zero_denominators_produce_zero_rates():
    result = evaluate_occupancy({"A": False}, {"A": False})

    assert result["tn"] == 1
    assert result["accuracy"] == 1.0
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_no_shared_labels_produces_no_evaluated_spots():
    result = evaluate_occupancy({"A": True}, {})

    assert result["evaluated_spots"] == 0
    assert result["prediction_only_ids"] == ["A"]
    assert all(result[name] == 0.0 for name in ("accuracy", "precision", "recall", "f1"))


@pytest.mark.parametrize(
    ("predicted", "ground_truth", "message"),
    [
        ({"A": 1}, {"A": True}, "Predicted"),
        ({"A": True}, {"A": None}, "Ground-truth"),
    ],
)
def test_non_boolean_values_raise(predicted, ground_truth, message):
    with pytest.raises(ValueError, match=message):
        evaluate_occupancy(predicted, ground_truth)


def test_micro_and_macro_are_distinct():
    first = evaluate_occupancy({"A": True}, {"A": True})
    second = evaluate_occupancy(
        {"B": True, "C": True, "D": True},
        {"B": False, "C": False, "D": False},
    )
    missing = evaluate_occupancy({}, {"E": True})

    summary = summarize_evaluations([first, second, missing])

    assert summary["image_count"] == 3
    assert summary["evaluated_image_count"] == 2
    assert summary["missing_prediction_count"] == 1
    assert summary["micro"]["evaluated_spots"] == 4
    assert summary["micro"]["tp"] == 1
    assert summary["micro"]["fp"] == 3
    assert summary["micro"]["accuracy"] == pytest.approx(0.25)
    assert summary["micro"]["precision"] == pytest.approx(0.25)
    assert summary["micro"]["recall"] == pytest.approx(1.0)
    assert summary["micro"]["f1"] == pytest.approx(0.4)
    assert summary["macro"]["accuracy"] == pytest.approx(0.5)
    assert summary["macro"]["precision"] == pytest.approx(0.5)
    assert summary["macro"]["recall"] == pytest.approx(0.5)
    assert summary["macro"]["f1"] == pytest.approx(0.5)


def test_empty_summary_uses_zero_rates():
    summary = summarize_evaluations([])

    assert summary["image_count"] == 0
    assert summary["evaluated_image_count"] == 0
    assert summary["micro"]["evaluated_spots"] == 0
    assert summary["micro"]["accuracy"] == 0.0
    assert summary["macro"]["accuracy"] == 0.0
