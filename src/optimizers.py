"""Long-only portfolio construction rules.

Every function takes a covariance matrix (and, where needed, a vector of expected
returns) estimated on a lookback window and returns weights that sum to one.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def ledoit_wolf_cov(returns):
    """Ledoit-Wolf shrinkage covariance, annualised (252 trading days)."""
    lw = LedoitWolf().fit(returns.values)
    return lw.covariance_ * 252


def sample_cov(returns):
    return returns.cov().values * 252


def _solve(objective, n, bounds=None):
    w0 = np.full(n, 1.0 / n)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    res = minimize(objective, w0, method="SLSQP", bounds=bounds or [(0.0, 1.0)] * n,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def equal_weight(cov, mu=None):
    n = cov.shape[0]
    return np.full(n, 1.0 / n)


def min_variance(cov, mu=None):
    return _solve(lambda w: w @ cov @ w, cov.shape[0])


def max_sharpe(cov, mu, rf=0.0, max_weight=0.30):
    """Tangency portfolio with a 30% cap per asset to limit estimation-error bets."""
    n = cov.shape[0]

    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mu - rf) / vol

    return _solve(neg_sharpe, n, bounds=[(0.0, max_weight)] * n)


def risk_parity(cov, mu=None):
    """Equal risk contribution: each asset contributes the same share of variance."""
    n = cov.shape[0]

    def objective(w):
        port_var = w @ cov @ w
        rc = w * (cov @ w) / port_var
        return ((rc - 1.0 / n) ** 2).sum()

    return _solve(objective, n, bounds=[(1e-6, 1.0)] * n)


def efficient_frontier(cov, mu, points=40):
    """Long-only frontier: minimum variance for a grid of target returns."""
    n = cov.shape[0]
    lo, hi = min_variance(cov) @ mu, mu.max()
    out = []
    for target in np.linspace(lo, hi * 0.999, points):
        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},
                {"type": "eq", "fun": lambda w, t=target: w @ mu - t})
        res = minimize(lambda w: w @ cov @ w, np.full(n, 1.0 / n), method="SLSQP",
                       bounds=[(0, 1)] * n, constraints=cons, options={"maxiter": 1000})
        if res.success:
            out.append((np.sqrt(res.x @ cov @ res.x), target))
    return np.array(out)


STRATEGIES = {
    "Equal weight": equal_weight,
    "Minimum variance": min_variance,
    "Max Sharpe (capped)": max_sharpe,
    "Risk parity": risk_parity,
}
