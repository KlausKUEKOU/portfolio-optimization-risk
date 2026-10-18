# Règles de construction de portefeuille (positions longues uniquement).
#
# Chaque fonction reçoit une matrice de covariance (et parfois les rendements attendus)
# estimée sur une fenêtre passée, et renvoie des poids positifs dont la somme vaut 1.

import numpy as np
from scipy.optimize import minimize


# ------------------------------------------------------------------------------
# ALGORITHME CovarianceLedoitWolf  (Ledoit et Wolf, 2004)
# Idée : la covariance empirique est bruitée quand on a beaucoup d'actifs.
#        On la "rétrécit" vers une cible simple : mu x Identité.
# ENTREE : R = matrice des rendements (n jours x p actifs)
# DEBUT
#   X <- R moins la moyenne de chaque colonne
#   S <- X'X / n                                  (covariance empirique)
#   mu <- trace(S) / p                            (variance moyenne)
#   delta <- distance entre S et mu x I           (à quel point S est loin de la cible)
#   beta <- bruit d'estimation de S               (calculé jour par jour)
#   alpha <- min(beta, delta) / delta             (intensité du rétrécissement, entre 0 et 1)
#   RETOURNER (1 - alpha) x S + alpha x mu x I, multiplié par 252 (annualisation)
# FIN
# ------------------------------------------------------------------------------
def ledoit_wolf_cov(returns):
    X = returns.values
    n, p = X.shape
    X = X - X.mean(axis=0)

    S = X.T @ X / n
    trace_by_asset = (X ** 2).sum(axis=0) / n
    mu = trace_by_asset.sum() / p

    sum_fourth = ((X ** 2).T @ (X ** 2)).sum()
    sum_sq_cov = ((X.T @ X) ** 2).sum() / n ** 2
    beta = (sum_fourth / n - sum_sq_cov) / (p * n)
    delta = (sum_sq_cov - 2 * mu * trace_by_asset.sum() + p * mu ** 2) / p
    if beta > delta:
        beta = delta
    if beta == 0:
        alpha = 0.0
    else:
        alpha = beta / delta

    shrunk = (1 - alpha) * S
    for i in range(p):
        shrunk[i, i] = shrunk[i, i] + alpha * mu
    return shrunk * 252


def portfolio_variance(w, cov):
    return w @ cov @ w


def solve(objective, n, bounds):
    """Minimise objective(w) sous les contraintes : somme des poids = 1 et bornes sur chaque poids.
    Le calcul est confié au solveur numérique SLSQP de scipy, en partant de poids égaux."""
    start = np.full(n, 1.0 / n)
    constraint = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    result = minimize(objective, start, method="SLSQP", bounds=bounds, constraints=(constraint,),
                      options={"maxiter": 1000, "ftol": 1e-12})
    w = result.x
    for i in range(n):                                  # nettoyage des petits négatifs numériques
        if w[i] < 0:
            w[i] = 0.0
    return w / w.sum()


# 1/N : le même poids pour tout le monde
def equal_weight(cov, mu=None):
    n = cov.shape[0]
    return np.full(n, 1.0 / n)


# Variance minimale : minimiser w' x Cov x w
def min_variance(cov, mu=None):
    n = cov.shape[0]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
    return solve(lambda w: portfolio_variance(w, cov), n, bounds)


# Ratio de Sharpe maximal, avec 30 % maximum par action pour limiter les paris sur des erreurs d'estimation
def max_sharpe(cov, mu, max_weight=0.30):
    n = cov.shape[0]

    def minus_sharpe(w):
        return -(w @ mu) / np.sqrt(portfolio_variance(w, cov))

    bounds = []
    for i in range(n):
        bounds.append((0.0, max_weight))
    return solve(minus_sharpe, n, bounds)


# ------------------------------------------------------------------------------
# ALGORITHME ParitéDeRisque
# Contribution au risque de l'actif i : RC_i = w_i x (Cov x w)_i / variance du portefeuille
# On cherche les poids tels que chaque RC_i = 1/n.
# On minimise donc la somme des écarts au carré (RC_i - 1/n)².
# ------------------------------------------------------------------------------
def risk_parity(cov, mu=None):
    n = cov.shape[0]

    def gap(w):
        total_var = portfolio_variance(w, cov)
        marginal = cov @ w
        error = 0.0
        for i in range(n):
            contribution = w[i] * marginal[i] / total_var
            error = error + (contribution - 1.0 / n) ** 2
        return error

    bounds = []
    for i in range(n):
        bounds.append((1e-6, 1.0))
    return solve(gap, n, bounds)


# ------------------------------------------------------------------------------
# ALGORITHME FrontièreEfficiente
# POUR chaque rendement cible entre celui du portefeuille de variance minimale et le maximum FAIRE
#     trouver les poids de variance minimale qui donnent exactement ce rendement
#     garder le point (volatilité, rendement)
# ------------------------------------------------------------------------------
def efficient_frontier(cov, mu, points=40):
    n = cov.shape[0]
    low = min_variance(cov) @ mu
    high = mu.max() * 0.999
    frontier = []
    for k in range(points):
        target = low + (high - low) * k / (points - 1)
        constraints = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},
                       {"type": "eq", "fun": lambda w, t=target: w @ mu - t})
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))
        result = minimize(lambda w: portfolio_variance(w, cov), np.full(n, 1.0 / n), method="SLSQP",
                          bounds=bounds, constraints=constraints, options={"maxiter": 1000})
        if result.success:
            frontier.append((np.sqrt(portfolio_variance(result.x, cov)), target))
    return np.array(frontier)


STRATEGIES = {
    "Equal weight": equal_weight,
    "Minimum variance": min_variance,
    "Max Sharpe (capped)": max_sharpe,
    "Risk parity": risk_parity,
}
