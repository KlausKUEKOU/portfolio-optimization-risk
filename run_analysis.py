"""Walk-forward portfolio backtest and VaR backtesting.

Usage: python run_analysis.py
Writes tables to results/ and charts to figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.optimizers import STRATEGIES, efficient_frontier, ledoit_wolf_cov
from src.risk import (VAR_METHODS, christoffersen_independence, expected_shortfall,
                      kupiec_pof, max_drawdown)

ROOT = Path(__file__).parent
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "figures").mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

LOOKBACK = 504          # two years of daily data to estimate risk
COST_BPS = 10           # transaction cost per unit of turnover
START = "2005-01-01"    # first rebalancing date
ALPHA = 0.99
VAR_WINDOW = 500

prices = pd.read_csv(ROOT / "data/stock_prices.csv", parse_dates=["date"], index_col="date")
returns = prices.pct_change(fill_method=None)

# ------------------------------------------------------------------ 1. frontier
window = returns.loc["2015-04-11":"2018-04-11"].dropna(axis=1)
cov = ledoit_wolf_cov(window)
mu = window.mean().values * 252
frontier = efficient_frontier(cov, mu)

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(frontier[:, 0], frontier[:, 1], color="#1f4e79", lw=2, label="Efficient frontier (long-only)")
asset_vol = np.sqrt(np.diag(cov))
ax.scatter(asset_vol, mu, s=18, color="grey", zorder=3)
for name, v, m in zip(window.columns, asset_vol, mu):
    ax.annotate(name, (v, m), fontsize=7, xytext=(3, 2), textcoords="offset points", color="grey")
for (label, fn), c in zip(STRATEGIES.items(), ["#7f7f7f", "#2ca02c", "#d62728", "#9467bd"]):
    w = fn(cov, mu)
    ax.scatter(np.sqrt(w @ cov @ w), w @ mu, s=70, color=c, edgecolor="black", zorder=4, label=label)
ax.set_xlabel("Annualised volatility")
ax.set_ylabel("Annualised expected return")
ax.set_title("Efficient frontier, April 2015 to April 2018 (Ledoit-Wolf covariance)")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(ROOT / "figures/efficient_frontier.png")
plt.close(fig)

# ------------------------------------------------------------------ 2. walk-forward backtest
rebal_dates = returns.loc[START:].resample("ME").last().index
rebal_dates = [returns.index[returns.index <= d][-1] for d in rebal_dates][:-1]

daily = {k: [] for k in STRATEGIES}
weights_hist = {k: {} for k in STRATEGIES}
turnover = {k: [] for k in STRATEGIES}
prev_w = {k: pd.Series(dtype=float) for k in STRATEGIES}

for i, d in enumerate(rebal_dates):
    hist = returns.loc[:d].iloc[-LOOKBACK:]
    hist = hist.dropna(axis=1)                       # assets with a full lookback only
    hist = hist.loc[:, prices.loc[d, hist.columns].notna()]
    cov = ledoit_wolf_cov(hist)
    mu = hist.mean().values * 252
    end = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else returns.index[-1]
    period = returns.loc[d:end].iloc[1:][hist.columns].fillna(0.0)
    for name, fn in STRATEGIES.items():
        w = pd.Series(fn(cov, mu), index=hist.columns)
        to = w.subtract(prev_w[name], fill_value=0).abs().sum()
        turnover[name].append(to)
        # buy-and-hold inside the month: weights drift with prices
        growth = (1 + period).cumprod()
        value = growth @ w
        port_r = value.pct_change()
        port_r.iloc[0] = value.iloc[0] - 1
        port_r.iloc[0] -= to * COST_BPS / 1e4
        daily[name].append(port_r)
        weights_hist[name][d] = w
        prev_w[name] = (growth.iloc[-1] * w) / value.iloc[-1]

port = pd.DataFrame({k: pd.concat(v) for k, v in daily.items()})
port.to_csv(ROOT / "results/strategy_daily_returns.csv")

rows = []
for name in port:
    r = port[name]
    wealth = (1 + r).cumprod()
    years = len(r) / 252
    cagr = wealth.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(252)
    rows.append({
        "Strategy": name,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe (rf=0)": r.mean() * 252 / vol,
        "Max drawdown": max_drawdown(wealth.values),
        "Worst day": r.min(),
        "Avg monthly turnover": np.mean(turnover[name][1:]),
        "99% 1-day hist. VaR": -np.quantile(r, 0.01),
        "99% 1-day ES": expected_shortfall(r.values, 0.99),
    })
metrics = pd.DataFrame(rows).set_index("Strategy")
metrics.to_csv(ROOT / "results/backtest_metrics.csv", float_format="%.4f")

colors = {"Equal weight": "#7f7f7f", "Minimum variance": "#2ca02c",
          "Max Sharpe (capped)": "#d62728", "Risk parity": "#9467bd"}
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
for name in port:
    wealth = (1 + port[name]).cumprod()
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

w_mv = pd.DataFrame(weights_hist["Minimum variance"]).T.fillna(0)
w_mv = w_mv.loc[:, w_mv.mean().sort_values(ascending=False).index]
top = w_mv.columns[:8]
w_plot = pd.concat([w_mv[top], w_mv.drop(columns=top).sum(axis=1).rename("Others")], axis=1)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.stackplot(w_plot.index, w_plot.T.values, labels=w_plot.columns, alpha=0.85)
ax.set_ylim(0, 1)
ax.set_title("Minimum-variance portfolio weights over time")
ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08))
fig.tight_layout()
fig.savefig(ROOT / "figures/min_variance_weights.png")
plt.close(fig)

# ------------------------------------------------------------------ 3. VaR backtest
r = port["Equal weight"].values
dates = port.index
records = {m: [] for m in VAR_METHODS}
for t in range(VAR_WINDOW, len(r)):
    sample = r[t - VAR_WINDOW:t]
    for m, fn in VAR_METHODS.items():
        if m.startswith("Monte"):
            records[m].append(fn(sample, ALPHA, n_sims=20_000, seed=t))
        else:
            records[m].append(fn(sample, ALPHA))
var_df = pd.DataFrame(records, index=dates[VAR_WINDOW:])
realised = pd.Series(r[VAR_WINDOW:], index=var_df.index)
var_df.assign(realised_return=realised).to_csv(ROOT / "results/var_forecasts.csv", float_format="%.6f")

bt_rows = []
for m in VAR_METHODS:
    viol = (realised < -var_df[m]).values
    rate, lr_pof, p_pof = kupiec_pof(viol, ALPHA)
    lr_ind, p_ind = christoffersen_independence(viol)
    bt_rows.append({"Method": m, "Days": len(viol), "Violations": int(viol.sum()),
                    "Expected": round(len(viol) * (1 - ALPHA), 1), "Violation rate": rate,
                    "Kupiec p-value": p_pof, "Independence p-value": p_ind,
                    "Avg VaR": var_df[m].mean()})
var_bt = pd.DataFrame(bt_rows).set_index("Method")
var_bt.to_csv(ROOT / "results/var_backtest.csv", float_format="%.4f")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(realised.index, realised, color="lightgrey", lw=0.6, label="Daily return")
for m, c in zip(VAR_METHODS, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]):
    ax.plot(var_df.index, -var_df[m], lw=1, color=c, label=f"-VaR {m}")
hist_viol = realised < -var_df["Historical"]
ax.scatter(realised.index[hist_viol], realised[hist_viol], s=8, color="black", zorder=3,
           label="Violation (historical VaR)")
ax.set_title("Rolling 99% 1-day VaR, equal-weight portfolio (500-day window)")
ax.legend(fontsize=7, ncol=3, loc="lower left")
fig.tight_layout()
fig.savefig(ROOT / "figures/var_backtest.png")
plt.close(fig)

pd.set_option("display.width", 160)
print(metrics.round(4))
print()
print(var_bt.round(4))
