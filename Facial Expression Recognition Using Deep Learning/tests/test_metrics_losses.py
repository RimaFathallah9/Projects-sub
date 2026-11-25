import numpy as np
import torch

from fer.evaluation.metrics import compute_metrics, expected_calibration_error
from fer.training.early_stopping import EarlyStopping
from fer.training.losses import FocalLoss, class_weights


def test_metrics_perfect_and_known_case():
    y = np.array([0, 1, 2, 3, 4, 5, 6])
    m = compute_metrics(y, y)
    assert m["accuracy"] == m["f1_macro"] == m["f1_weighted"] == 1.0
    # 2 classes present; class 0: 1/2 recall, class 1: perfect. Other classes absent -> 0 under macro.
    yt, yp = np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])
    m = compute_metrics(yt, yp)
    assert m["accuracy"] == 0.75
    assert abs(m["per_class"]["angry"]["recall"] - 0.5) < 1e-9
    assert abs(m["per_class"]["disgust"]["precision"] - 2 / 3) < 1e-9
    assert m["confusion_matrix"][0][:2] == [1, 1]
    assert abs(m["f1_macro"] - (2 / 3 + 0.8) / 7) < 1e-9  # absent classes count as 0 (zero_division=0)


def test_ece_zero_for_calibrated_confident_correct():
    probs = np.eye(7)[[0, 1, 2]]
    assert expected_calibration_error(probs, np.array([0, 1, 2])) == 0.0


def test_class_weights_properties():
    c = [100, 10, 50, 400, 200, 100, 140]
    for s in ("inverse", "sqrt_inverse", "effective_num"):
        w = class_weights(c, s)
        assert abs(w.sum() - 7) < 1e-9
        assert w[1] == w.max() and w[3] == w.min()
    assert class_weights(c, "none") is None
    inv, sq = class_weights(c, "inverse"), class_weights(c, "sqrt_inverse")
    assert inv.max() / inv.min() > sq.max() / sq.min()


def test_focal_matches_ce_at_gamma_zero():
    logits, y = torch.randn(16, 7), torch.randint(0, 7, (16,))
    ce = torch.nn.functional.cross_entropy(logits, y)
    assert torch.allclose(FocalLoss(0.0)(logits, y), ce, atol=1e-6)
    # (1 - p_t)^gamma <= 1, so focal loss is never larger than cross-entropy.
    assert FocalLoss(2.0)(logits, y) <= ce + 1e-5


def test_early_stopping_max_mode():
    es = EarlyStopping(patience=2, mode="max")
    assert es.step(0.5, 1) and es.step(0.6, 2)
    assert not es.step(0.6, 3) and not es.should_stop
    assert not es.step(0.55, 4) and es.should_stop
    assert es.best_epoch == 2
