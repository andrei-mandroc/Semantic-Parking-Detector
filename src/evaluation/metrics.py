"""Parking-space occupancy metrics with occupied as the positive class."""

from __future__ import annotations


RATE_NAMES = ("accuracy", "precision", "recall", "f1")
COUNT_NAMES = ("tp", "tn", "fp", "fn")


def _rates(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_occupancy(
    predicted: dict[str, bool], ground_truth: dict[str, bool]
) -> dict:
    """Compare shared space IDs and report IDs excluded from scoring.

    Missing predictions and predictions without a valid reference label are
    excluded. Rates with a zero denominator are 0.0. Duplicate IDs cannot be
    detected after input is converted to dictionaries and must be rejected
    while parsing annotations and parking-space geometry.
    """
    if any(type(value) is not bool for value in predicted.values()):
        raise ValueError("Predicted occupancy values must be bool")
    if any(type(value) is not bool for value in ground_truth.values()):
        raise ValueError("Ground-truth occupancy values must be bool")

    shared_ids = predicted.keys() & ground_truth.keys()
    counts = {name: 0 for name in COUNT_NAMES}
    for space_id in shared_ids:
        prediction = predicted[space_id]
        label = ground_truth[space_id]
        if prediction and label:
            counts["tp"] += 1
        elif prediction and not label:
            counts["fp"] += 1
        elif not prediction and label:
            counts["fn"] += 1
        else:
            counts["tn"] += 1

    return {
        **counts,
        "evaluated_spots": len(shared_ids),
        "missing_prediction_ids": sorted(ground_truth.keys() - predicted.keys()),
        "prediction_only_ids": sorted(predicted.keys() - ground_truth.keys()),
        **_rates(**counts),
    }


def summarize_evaluations(image_results: list[dict]) -> dict:
    """Return global spot-level rates and mean rates over evaluated images."""
    counts = {name: sum(result[name] for result in image_results) for name in COUNT_NAMES}
    evaluated_results = [result for result in image_results if result["evaluated_spots"]]
    macro = {
        name: (
            sum(result[name] for result in evaluated_results) / len(evaluated_results)
            if evaluated_results else 0.0
        )
        for name in RATE_NAMES
    }
    return {
        "image_count": len(image_results),
        "evaluated_image_count": len(evaluated_results),
        "missing_prediction_count": sum(
            len(result["missing_prediction_ids"]) for result in image_results
        ),
        "prediction_only_count": sum(
            len(result["prediction_only_ids"]) for result in image_results
        ),
        "micro": {
            **counts,
            "evaluated_spots": sum(counts.values()),
            **_rates(**counts),
        },
        "macro": macro,
    }
