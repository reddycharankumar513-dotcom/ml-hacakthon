"""
Unit tests for macro F0.5 evaluator and edge cases.
"""

import pytest
from business_entity_resolution.evaluation import compute_entity_f05, evaluate_macro_f05

def test_singleton_correct_score():
    res = compute_entity_f05(set(), set())
    assert res["f0_5"] == 1.0
    assert res["is_singleton"] is True
    assert res["correct_singleton"] is True

def test_singleton_false_positive_score():
    res = compute_entity_f05({"S2-00001"}, set())
    assert res["f0_5"] == 0.0
    assert res["is_singleton"] is True
    assert res["correct_singleton"] is False

def test_perfect_match_score():
    gt = {"S2-00001", "S3-00002"}
    preds = {"S2-00001", "S3-00002"}
    res = compute_entity_f05(preds, gt)
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f0_5"] == 1.0

def test_precision_heavy_f05():
    # 2 predicted, 1 true match -> Precision=0.5, Recall=1.0
    gt = {"S2-00001"}
    preds = {"S2-00001", "S2-00002"}
    res = compute_entity_f05(preds, gt)
    # F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 0.5555...
    assert pytest.approx(res["f0_5"], 0.001) == 0.5555

def test_macro_evaluation():
    gt_dict = {
        "S1-1": {"S2-A"},
        "S1-2": set()
    }
    pred_dict = {
        "S1-1": {"S2-A"},
        "S1-2": set()
    }
    metrics = evaluate_macro_f05(pred_dict, gt_dict, ["S1-1", "S1-2"])
    assert metrics["macro_f0_5"] == 1.0
    assert metrics["singleton_accuracy"] == 1.0
