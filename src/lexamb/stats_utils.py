"""Statistical machinery (named stats_utils to avoid shadowing the stdlib ``statistics`` module).

Probe estimator
---------------
Ridge regression of a target y on a standardised layer state x with *unpenalised* covariates C
(intercept always included):

    min_{b, g} ||y - C b - X g||^2 + alpha ||g||^2

By the Frisch-Waugh-Lovell theorem this equals ridge of M_C y on M_C X (M_C = I - P_C), so the
covariates are fully partialled out and only the hidden state is shrunk. We solve it in the dual
(kernel) form, K = X X^T, which is exact and fast when n << d (here n <= 1.6k, d up to 4096).

alpha is chosen per target by exact leave-one-out error, using the closed-form hat diagonal
h_ii = [P_C]_ii + sum_j U_ij^2 s_j/(s_j+alpha) (Golub, Heath & Wahba 1979; Hastie et al. ESL §7.10).
The grid is relative to the mean kernel eigenvalue, making it scale-invariant across layers/models.

With C = [1] this is an ordinary ridge probe whose covariate-only baseline is the training mean.
With C = controls it is a *conditional probe* (Hewitt et al., EMNLP 2021): the increment
R^2(C + h) - R^2(C) estimates information in the layer beyond the lexical covariates.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


# ============================================================================ kernel ridge
def fwl_kernel_ridge(K_TT: np.ndarray, K_ET: np.ndarray, C_T: np.ndarray, C_E: np.ndarray,
                     Y_T: np.ndarray, lambdas_rel: np.ndarray):
    """Fit on training rows T, predict test rows E, for many target columns at once.

    Returns pred_E (nE, m), loo_T (nT, m) leave-one-out fitted values, cov_E (nE, m) covariate-only
    test predictions, chosen alpha index (m,).
    """
    nT = K_TT.shape[0]
    Q, _ = np.linalg.qr(C_T)
    pinvC = np.linalg.pinv(C_T)                       # (k, nT)
    b_y = pinvC @ Y_T                                 # covariate coefficients (OLS)
    Yr = Y_T - C_T @ b_y                              # M_C y
    KQ = K_TT @ Q
    Kr = K_TT - Q @ KQ.T - KQ @ Q.T + Q @ (Q.T @ KQ) @ Q.T  # M K M
    Kr = (Kr + Kr.T) / 2
    s, U = np.linalg.eigh(Kr)
    s = np.clip(s, 0, None)
    scale = max(s.sum() / max(nT - Q.shape[1], 1), 1e-12)
    alphas = np.asarray(lambdas_rel) * scale
    UtY = U.T @ Yr                                    # (nT, m)
    hP = np.sum(Q ** 2, axis=1)                       # diag(P_C)
    U2 = U ** 2
    best_err = np.full(Y_T.shape[1], np.inf)
    best_idx = np.zeros(Y_T.shape[1], dtype=int)
    best_loo = np.zeros_like(Y_T)
    for a_i, a in enumerate(alphas):
        f = s / (s + a)
        h = hP + U2 @ f
        fit_r = U @ (f[:, None] * UtY)
        loo_res = (Yr - fit_r) / np.clip(1 - h, 1e-8, None)[:, None]
        err = np.mean(loo_res ** 2, axis=0)
        better = err < best_err
        if better.any():
            best_err[better] = err[better]
            best_idx[better] = a_i
            best_loo[:, better] = (Y_T - loo_res)[:, better]
    # dual coefficients with each column's own alpha
    inv = 1.0 / (s[:, None] + alphas[best_idx][None, :])
    A = U @ (inv * UtY)                                # (nT, m)
    # test kernel against residualised training states: (K_ET - C_E pinvC K_TT) M
    K_Er = K_ET - C_E @ (pinvC @ K_TT)
    K_Er = K_Er - (K_Er @ Q) @ Q.T
    cov_E = C_E @ b_y
    pred_E = cov_E + K_Er @ A
    return pred_E, best_loo, cov_E, best_idx


def standardise_fit(X: np.ndarray):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def platt(train_scores: np.ndarray, train_y01: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """Platt (1999) scaling fitted on held-out (LOO) training decision values only."""
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(train_y01)) < 2:
        return np.full(len(test_scores), train_y01.mean())
    lr = LogisticRegression(C=1e4, max_iter=1000)
    lr.fit(train_scores.reshape(-1, 1), train_y01)
    return lr.predict_proba(test_scores.reshape(-1, 1))[:, 1]


# ============================================================================ metrics
def r2(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Out-of-sample R^2 (can be negative) along the last axis."""
    ss_res = np.sum((y - p) ** 2, axis=-1)
    ss_tot = np.sum((y - y.mean(axis=-1, keepdims=True)) ** 2, axis=-1)
    return 1 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)


def mae(y, p):
    return np.mean(np.abs(y - p), axis=-1)


def spearman(y, p):
    if np.std(p) == 0 or np.std(y) == 0:
        return np.nan
    return sps.spearmanr(y, p).statistic


def auroc(y01: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Rank-based AUROC (Mann-Whitney), vectorised over leading axes of s; y01 broadcastable."""
    ranks = sps.rankdata(s, axis=-1)
    pos = (y01 == 1)
    n1 = pos.sum(axis=-1)
    n0 = (~pos).sum(axis=-1)
    rs = np.sum(ranks * pos, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rs - n1 * (n1 + 1) / 2) / (n1 * n0)


def balanced_acc(y01, prob):
    pred = prob >= 0.5
    tpr = np.mean(pred[y01 == 1]) if np.any(y01 == 1) else np.nan
    tnr = np.mean(~pred[y01 == 0]) if np.any(y01 == 0) else np.nan
    return (tpr + tnr) / 2


def macro_f1(y01, prob):
    from sklearn.metrics import f1_score
    return f1_score(y01.astype(int), (prob >= 0.5).astype(int), average="macro", zero_division=0)


def brier(y01, prob):
    return np.mean((prob - y01) ** 2)


# ============================================================================ inference
def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg (1995) adjusted p-values; NaNs are left out of the family."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    if m == 0:
        return q
    ps = p[ok]
    order = np.argsort(ps)
    ranked = ps[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def perm_pvalue(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """One-sided permutation p, (1 + #{null >= obs}) / (1 + P) (Phipson & Smyth 2010). null: (P, ...)."""
    return (1 + np.sum(null >= obs[None], axis=0)) / (1 + null.shape[0])


def maxT_pvalue(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Westfall-Young max-statistic FWER-adjusted p across layers (same permutations for all layers)."""
    mx = np.nanmax(null, axis=1)
    return (1 + np.sum(mx[:, None] >= obs[None, :], axis=0)) / (1 + null.shape[0])


def boot_p_two_sided(boot: np.ndarray) -> np.ndarray:
    """Two-sided bootstrap p for H0: parameter = 0 (percentile inversion)."""
    lo = np.mean(boot <= 0, axis=0)
    hi = np.mean(boot >= 0, axis=0)
    return np.clip(2 * np.minimum(lo, hi), 1.0 / boot.shape[0], 1.0)


def ols_hc3(y: np.ndarray, X: np.ndarray, names: list[str]):
    """OLS with HC3 heteroscedasticity-consistent SEs (MacKinnon & White 1985)."""
    import statsmodels.api as sm
    res = sm.OLS(y, X).fit(cov_type="HC3")
    ci = res.conf_int()
    return {n: {"beta": float(res.params[i]), "se": float(res.bse[i]), "p": float(res.pvalues[i]),
                "ci_lo": float(ci[i, 0]), "ci_hi": float(ci[i, 1])} for i, n in enumerate(names)}, res


def vif(X: np.ndarray) -> np.ndarray:
    out = []
    for j in range(X.shape[1]):
        others = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])
        beta, *_ = np.linalg.lstsq(others, X[:, j], rcond=None)
        res = X[:, j] - others @ beta
        r2_ = 1 - res.var() / max(X[:, j].var(), 1e-12)
        out.append(1 / max(1 - r2_, 1e-12))
    return np.array(out)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al., ICML 2019) via centred Gram matrices."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    Kx, Ky = X @ X.T, Y @ Y.T
    hsic = np.sum(Kx * Ky)
    return float(hsic / np.sqrt(np.sum(Kx * Kx) * np.sum(Ky * Ky)))


def centre_of_mass_depth(depth: np.ndarray, score: np.ndarray) -> float:
    """Decodability-weighted mean relative depth, sum d*max(s,0) / sum max(s,0)."""
    w = np.clip(score, 0, None)
    return float(np.sum(depth * w) / np.sum(w)) if np.sum(w) > 0 else np.nan


def expected_layer_tenney(depth: np.ndarray, score: np.ndarray) -> float:
    """Tenney et al. (ACL 2019) expected layer from differential scores; increments clipped at 0
    so that the estimate stays inside [0, 1] when later layers lose information."""
    inc = np.clip(np.diff(score), 0, None)
    return float(np.sum(depth[1:] * inc) / np.sum(inc)) if np.sum(inc) > 0 else np.nan
