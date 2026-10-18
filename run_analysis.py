# Backtest "walk-forward" de 4 portefeuilles et validation de 5 modèles de VaR.
#
# Lancement : python run_analysis.py   (environ 3 minutes)
# Sorties   : tableaux dans results/ et graphiques dans figures/

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.optimizers import STRATEGIES, efficient_frontier, ledoit_wolf_cov
from src.risk import (VAR_METHODS, christoffersen_independence, expected_shortfall, kupiec_pof,
                      max_drawdown, quantile)

ROOT = Path(__file__).parent
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "figures").mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

LOOKBACK = 504          # 2 ans de jours de bourse pour estimer le risque
COST_BPS = 10           # coût de transaction : 0,10 % du montant échangé
START = "2005-01-01"    # première date de rééquilibrage
ALPHA = 0.99            # niveau de la VaR
VAR_WINDOW = 500        # fenêtre glissante pour la VaR

prices = pd.read_csv(ROOT / "data/stock_prices.csv", parse_dates=["date"], index_col="date")
returns = prices.pct_change(fill_method=None)          # rendement du jour = P(t) / P(t-1) - 1

# ==================================================================== 1. FRONTIÈRE EFFICIENTE
window = returns.loc["2015-04-11":"2018-04-11"].dropna(axis=1)
cov = ledoit_wolf_cov(window)
mu = window.mean().values * 252
frontier = efficient_frontier(cov, mu)

colors = {"Equal weight": "#7f7f7f", "Minimum variance": "#2ca02c",
          "Max Sharpe (capped)": "#d62728", "Risk parity": "#9467bd"}
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(frontier[:, 0], frontier[:, 1], color="#1f4e79", lw=2, label="Efficient frontier (long-only)")
for k in range(len(window.columns)):
    vol_k = np.sqrt(cov[k, k])
    ax.scatter(vol_k, mu[k], s=18, color="grey", zorder=3)
    ax.annotate(window.columns[k], (vol_k, mu[k]), fontsize=7, xytext=(3, 2), textcoords="offset points",
                color="grey")
for name in STRATEGIES:
    w = STRATEGIES[name](cov, mu)
    ax.scatter(np.sqrt(w @ cov @ w), w @ mu, s=70, color=colors[name], edgecolor="black", zorder=4, label=name)
ax.set_xlabel("Annualised volatility")
ax.set_ylabel("Annualised expected return")
ax.set_title("Efficient frontier, April 2015 to April 2018 (Ledoit-Wolf covariance)")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(ROOT / "figures/efficient_frontier.png")
plt.close(fig)

# ==================================================================== 2. BACKTEST
# ALGORITHME BacktestWalkForward
# POUR chaque fin de mois d FAIRE
#     historique <- les 504 derniers jours avant d
#     garder les actions qui ont un historique complet et un prix à la date d
#     Cov <- Ledoit-Wolf(historique) ; mu <- rendement moyen annualisé
#     POUR chaque stratégie FAIRE
#         w <- poids de la stratégie
#         rotation <- somme des |w - poids de la veille|  ; coût <- rotation x 0,10 %
#         pendant le mois suivant, les poids dérivent avec les prix (pas d'achat/vente)
#         enregistrer les rendements journaliers du portefeuille
#     FIN POUR
# FIN POUR
# On n'utilise jamais une donnée postérieure à d : c'est un test hors échantillon.

all_month_ends = returns.loc[START:].resample("ME").last().index
rebal_dates = []
for d in all_month_ends:
    trading_days_before = returns.index[returns.index <= d]
    rebal_dates.append(trading_days_before[-1])          # dernier jour de bourse du mois
rebal_dates = rebal_dates[:-1]

daily_returns = {}
weights_history = {}
turnover = {}
previous_weights = {}
for name in STRATEGIES:
    daily_returns[name] = []
    weights_history[name] = {}
    turnover[name] = []
    previous_weights[name] = pd.Series(dtype=float)

for i in range(len(rebal_dates)):
    d = rebal_dates[i]

    history = returns.loc[:d].iloc[-LOOKBACK:]
    history = history.dropna(axis=1)
    kept = []
    for asset in history.columns:
        if not pd.isna(prices.loc[d, asset]):
            kept.append(asset)
    history = history[kept]

    cov = ledoit_wolf_cov(history)
    mu = history.mean().values * 252

    if i + 1 < len(rebal_dates):
        end = rebal_dates[i + 1]
    else:
        end = returns.index[-1]
    next_month = returns.loc[d:end].iloc[1:][kept].fillna(0.0)
    growth = (1 + next_month).cumprod()                 # valeur de 1 € placé dans chaque action

    for name in STRATEGIES:
        w = pd.Series(STRATEGIES[name](cov, mu), index=kept)

        change = w.subtract(previous_weights[name], fill_value=0)
        rotation = 0.0
        for x in change:
            rotation = rotation + abs(x)
        turnover[name].append(rotation)

        value = growth @ w                              # valeur du portefeuille chaque jour du mois
        port_r = value.pct_change()
        port_r.iloc[0] = value.iloc[0] - 1
        port_r.iloc[0] = port_r.iloc[0] - rotation * COST_BPS / 10000
        daily_returns[name].append(port_r)
        weights_history[name][d] = w
        previous_weights[name] = growth.iloc[-1] * w / value.iloc[-1]   # poids après dérive

portfolio = pd.DataFrame()
for name in STRATEGIES:
    portfolio[name] = pd.concat(daily_returns[name])
portfolio.to_csv(ROOT / "results/strategy_daily_returns.csv")

rows = []
for name in STRATEGIES:
    r = list(portfolio[name].values)
    wealth = []
    level = 1.0
    for x in r:
        level = level * (1 + x)
        wealth.append(level)
    years = len(r) / 252
    cagr = wealth[-1] ** (1 / years) - 1
    vol = np.std(r, ddof=1) * np.sqrt(252)
    rows.append({
        "Strategy": name,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe (rf=0)": np.mean(r) * 252 / vol,
        "Max drawdown": max_drawdown(wealth),
        "Worst day": min(r),
        "Avg monthly turnover": np.mean(turnover[name][1:]),    # le 1er mois = achat initial, exclu
        "99% 1-day hist. VaR": -quantile(r, 0.01),
        "99% 1-day ES": expected_shortfall(r, 0.99),
    })
metrics = pd.DataFrame(rows).set_index("Strategy")
metrics.to_csv(ROOT / "results/backtest_metrics.csv", float_format="%.4f")

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
for name in STRATEGIES:
    wealth = (1 + portfolio[name]).cumprod()
    axes[0].plot(wealth, label=name, color=colors[name], lw=1.4)
    axes[1].plot(wealth / wealth.cummax() - 1, color=colors[name], lw=1)
axes[0].set_yscale("log")
axes[0].set_ylabel("Growth of $1 (log scale)")
axes[0].set_title("Out-of-sample walk-forward backtest, monthly rebalancing, 10 bps costs")
axes[0].legend(fontsize=8)
axes[1].set_ylabel("Drawdown")
fig.tight_layout()
fig.savefig(ROOT / "figures/backtest_wealth_drawdown.png")
plt.close(fig)

# poids du portefeuille de variance minimale : les 8 plus gros + "Others"
w_mv = pd.DataFrame(weights_history["Minimum variance"]).T.fillna(0)
average_weight = w_mv.mean().sort_values(ascending=False)
top = list(average_weight.index[:8])
others = w_mv.drop(columns=top).sum(axis=1)
to_plot = w_mv[top].copy()
to_plot["Others"] = others
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.stackplot(to_plot.index, to_plot.T.values, labels=to_plot.columns, alpha=0.85)
ax.set_ylim(0, 1)
ax.set_title("Minimum-variance portfolio weights over time")
ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08))
fig.tight_layout()
fig.savefig(ROOT / "figures/min_variance_weights.png")
plt.close(fig)

# ==================================================================== 3. VALIDATION DE LA VaR
# ALGORITHME BacktestVaR
# POUR chaque jour t après les 500 premiers FAIRE
#     échantillon <- les 500 rendements avant t
#     POUR chaque méthode : prévision[t] <- VaR(échantillon)
#     dépassement[t] <- (rendement réel du jour t < -prévision[t])
# Puis tests de Kupiec (fréquence) et de Christoffersen (regroupement) sur les dépassements.
r = list(portfolio["Equal weight"].values)
dates = portfolio.index
forecasts = {}
for method in VAR_METHODS:
    forecasts[method] = []

for t in range(VAR_WINDOW, len(r)):
    sample = r[t - VAR_WINDOW:t]
    for method in VAR_METHODS:
        if method.startswith("Monte"):
            forecasts[method].append(VAR_METHODS[method](sample, ALPHA, n_sims=20_000, seed=t))
        else:
            forecasts[method].append(VAR_METHODS[method](sample, ALPHA))

realised = r[VAR_WINDOW:]
test_dates = dates[VAR_WINDOW:]
table = pd.DataFrame(forecasts, index=test_dates)
table["realised_return"] = realised
table.to_csv(ROOT / "results/var_forecasts.csv", float_format="%.6f")

backtest_rows = []
for method in VAR_METHODS:
    violations = []
    for t in range(len(realised)):
        violations.append(realised[t] < -forecasts[method][t])
    rate, lr, p_kupiec = kupiec_pof(violations, ALPHA)
    lr_ind, p_ind = christoffersen_independence(violations)
    backtest_rows.append({"Method": method, "Days": len(violations), "Violations": sum(violations),
                          "Expected": round(len(violations) * (1 - ALPHA), 1), "Violation rate": rate,
                          "Kupiec p-value": p_kupiec, "Independence p-value": p_ind,
                          "Avg VaR": np.mean(forecasts[method])})
var_backtest = pd.DataFrame(backtest_rows).set_index("Method")
var_backtest.to_csv(ROOT / "results/var_backtest.csv", float_format="%.4f")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(test_dates, realised, color="lightgrey", lw=0.6, label="Daily return")
line_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
k = 0
for method in VAR_METHODS:
    ax.plot(test_dates, -np.array(forecasts[method]), lw=1, color=line_colors[k], label="-VaR " + method)
    k = k + 1
hit_dates, hit_values = [], []
for t in range(len(realised)):
    if realised[t] < -forecasts["Historical"][t]:
        hit_dates.append(test_dates[t])
        hit_values.append(realised[t])
ax.scatter(hit_dates, hit_values, s=8, color="black", zorder=3, label="Violation (historical VaR)")
ax.set_title("Rolling 99% 1-day VaR, equal-weight portfolio (500-day window)")
ax.legend(fontsize=7, ncol=3, loc="lower left")
fig.tight_layout()
fig.savefig(ROOT / "figures/var_backtest.png")
plt.close(fig)

print(metrics.round(4))
print(var_backtest.round(4))
