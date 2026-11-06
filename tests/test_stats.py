"""Unit tests for the statistics that support every comparative claim.

Run:  python -m pytest -q tests     (NumPy/SciPy/scikit-learn only; no GPU)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from drxai import ensemble as E
from drxai import metrics as M
from drxai import splits as SP
from drxai import stats as S

rng = np.random.default_rng(0)


def _probs(y, k=5, signal=1.5, seed=0):
    r = np.random.default_rng(seed)
    logits = r.normal(size=(len(y), k))
    logits[np.arange(len(y)), y] += signal
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def test_auc_matches_sklearn():
    y = rng.integers(0, 5, 400)
    P = _probs(y)
    for c in range(5):
        assert np.isclose(M.per_class_auc(y, P)[c], roc_auc_score(y == c, P[:, c]))


def test_qwk_matches_sklearn():
    y = rng.integers(0, 5, 500)
    yhat = np.clip(y + rng.integers(-1, 2, 500), 0, 4)
    assert np.isclose(M.quadratic_weighted_kappa(y, yhat),
                      cohen_kappa_score(y, yhat, weights="quadratic"))


def test_generalised_delong_reduces_to_delong_binary():
    """Proposition 2: for a single AUC the estimator equals DeLong (1988)."""
    y = rng.integers(0, 2, 300)
    a = y + rng.normal(0, 1.0, 300)
    b = y + rng.normal(0, 1.3, 300)
    P_a = np.c_[1 - a, a]
    P_b = np.c_[1 - b, b]
    g = S.delong_macro(y, P_a, P_b, classes=[1])
    d = S.delong_binary(y, a, b)
    assert np.isclose(g["z"], d["z"], rtol=1e-10)
    assert np.isclose(g["p"], d["p"], rtol=1e-10)


def test_generalised_delong_type1_error():
    """Under H0 (two exchangeable noisy copies) the test rejects ~5% of the time."""
    rej = 0
    for s in range(300):
        r = np.random.default_rng(100 + s)
        y = r.integers(0, 5, 250)
        base = r.normal(size=(250, 5)); base[np.arange(250), y] += 1.0
        Pa = base + r.normal(0, 0.7, base.shape)
        Pb = base + r.normal(0, 0.7, base.shape)
        rej += S.delong_macro(y, Pa, Pb)["p"] < 0.05
    assert 0.02 <= rej / 300 <= 0.09


def test_generalised_delong_variance_matches_bootstrap():
    y = rng.integers(0, 5, 600)
    Pa, Pb = _probs(y, signal=1.6, seed=1), _probs(y, signal=1.3, seed=2)
    se = S.delong_macro(y, Pa, Pb)["se"]
    reps = []
    for b in range(400):
        i = np.concatenate([rng.choice(np.where(y == c)[0], (y == c).sum()) for c in range(5)])
        reps.append(M.macro_auc(y[i], Pa[i]) - M.macro_auc(y[i], Pb[i]))
    assert abs(np.std(reps) / se - 1) < 0.2


def test_holm():
    assert np.allclose(S.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_mcnemar_exact_symmetric():
    a = np.array([1] * 5 + [0] * 5 + [1] * 50, bool)
    b = np.array([0] * 5 + [1] * 5 + [1] * 50, bool)
    assert S.mcnemar(a, b)["p"] == 1.0


def test_bootstrap_ci_is_bounded():
    y = rng.integers(0, 5, 300)
    P = _probs(y, signal=4.0)
    groups = np.repeat(np.arange(150), 2)
    ci = S.cluster_bootstrap(lambda i: float((P[i].argmax(1) == y[i]).mean()), groups, y, 200)
    assert 0 <= ci["lo"] <= ci["estimate"] <= ci["hi"] <= 1


def test_proposition1_bound_holds():
    aucs = np.array([0.872, 0.895, 0.903])
    y = rng.integers(0, 5, 500)
    probs = np.stack([_probs(y, seed=s) for s in range(3)])
    w = E.auc_weights(aucs)
    diff = np.abs(E.fuse(probs, "auc_weighted", w) - E.fuse(probs, "mean")).max()
    b = E.uniform_bound(aucs)
    assert diff <= b["bound_delta"] + 1e-12 <= b["bound_range"] + 1e-12


def test_patient_grouped_folds_do_not_leak():
    n = 1000
    df = pd.DataFrame({"group": np.repeat(np.arange(n // 2), 2),
                       "grade": np.repeat(rng.integers(0, 5, n // 2), 2)})
    fold = SP.assign_folds(df, 5, 42)
    for k in range(5):
        SP.check_no_leakage(df, SP.iteration_roles(fold, k))
    assert set(fold) == set(range(5))


def test_ece_perfect_calibration_is_small():
    r = np.random.default_rng(3)
    conf = r.uniform(0.2, 1, 20000)
    y = (r.uniform(size=20000) < conf).astype(int)
    P = np.c_[conf, 1 - conf]
    yy = np.where(y == 1, 0, 1)            # class 0 correct with probability conf
    assert M.expected_calibration_error(yy, P) < 0.02


def test_ppv_npv():
    ppv, npv = M.predictive_values(0.934, 0.952, 0.20)
    assert round(ppv, 3) == 0.829 and round(npv, 3) == 0.983


def test_clustered_delong_equals_unclustered_for_singletons():
    y = rng.integers(0, 5, 300)
    Pa, Pb = _probs(y, seed=4), _probs(y, seed=5)
    u = S.delong_macro(y, Pa, Pb)
    c = S.delong_macro(y, Pa, Pb, groups=np.arange(300))
    assert np.isclose(u["se"], c["se"])


def test_clustered_delong_type1_with_paired_eyes():
    """Fellow eyes share a latent severity/noise; clustering keeps size near 5%."""
    rej = 0
    for s in range(200):
        r = np.random.default_rng(500 + s)
        g = r.integers(0, 5, 150); y = np.repeat(g, 2); grp = np.repeat(np.arange(150), 2)
        shared = np.repeat(r.normal(size=(150, 5)), 2, axis=0)
        base = shared.copy(); base[np.arange(300), y] += 1.0
        Pa = base + r.normal(0, .5, base.shape); Pb = base + r.normal(0, .5, base.shape)
        rej += S.delong_macro(y, Pa, Pb, groups=grp)["p"] < 0.05
    assert rej / 200 <= 0.09
