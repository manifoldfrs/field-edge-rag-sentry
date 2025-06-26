import json
from pathlib import Path

from ragas.metrics import answer_correctness, context_precision, faithfulness
from ragas.schema import Dataset, GroundTruth

EVAL_PATH = Path("data/eval_set.jsonl").resolve()


def _dummy_dataset():
    # minimal 1‑row dataset to keep test light‑weight
    gt = GroundTruth(questions=["dummy question"], answers=["ground truth answer"])
    ds = Dataset.from_groundtruth(gt)
    ds.predicted_answers = ["ground truth answer"]
    ds.contexts = ["supporting context"]
    return ds


def test_metric_ranges():
    ds = _dummy_dataset()
    for metric in [context_precision, faithfulness, answer_correctness]:
        score = metric.aggregate(metric.score(ds))
        assert 0.0 <= score <= 1.0, f"{metric.name} out of range"
