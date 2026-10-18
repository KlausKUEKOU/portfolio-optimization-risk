# Portfolio Optimization and Market Risk

Do optimized portfolios actually beat naive diversification **out of sample**, and do standard
Value-at-Risk models survive a crisis? This project answers both questions with a walk-forward
backtest on 20 US large caps (2005 to 2018, including the 2008 crash).

![Backtest](figures/backtest_wealth_drawdown.png)

## Method

**Portfolio construction** (long-only, rebalanced monthly, 10 bps transaction costs):

| Strategy | Rule |
|---|---|
| Equal weight | 1/N, the naive benchmark (DeMiguel, Garlappi and Uppal, 2009) |
| Minimum variance | Minimise w'Σw |
| Max Sharpe (capped) | Tangency portfolio, 30% cap per stock to limit estimation-error bets |
| Risk parity | Each stock contributes the same share of portfolio variance |

- Σ is estimated with **Ledoit-Wolf shrinkage** on a rolling 2-year window; only stocks with a full
  window enter the universe (no look-ahead: e.g. FB enters in 2014, BABA in 2016).
- Weights drift with prices between rebalancing dates; turnover is charged at each rebalance.

**Risk models** (1-day 99% VaR on the equal-weight portfolio, rolling 500-day window):
historical simulation, Gaussian, Cornish-Fisher, Monte Carlo Student-t and EWMA (RiskMetrics,
λ = 0.94). Each is backtested with the **Kupiec** proportion-of-failures test and the
**Christoffersen** independence test.

## Results

### Out-of-sample performance, Feb 2005 to Apr 2018

| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Monthly turnover | 99% ES (1 day) |
|---|---|---|---|---|---|---|
| Equal weight | 12.6% | 22.6% | 0.64 | -56.5% | 6% | 5.8% |
| Minimum variance | 8.5% | **16.2%** | 0.58 | **-38.9%** | 9% | **4.2%** |
| Max Sharpe (capped) | **16.7%** | 24.3% | **0.76** | -56.5% | 32% | 5.9% |
| Risk parity | 12.1% | 20.6% | 0.66 | -52.8% | 6% | 5.4% |

- **Minimum variance delivers what it promises:** 28% less volatility and a drawdown 17 points
  shallower than 1/N, at the cost of return.
- **Max Sharpe has the best Sharpe here, but trades 5 times more** and is the most exposed to
  estimation error: it concentrates on the recent winners (on average 17% in AAPL, 12% in SBUX,
  10% in AMZN), a momentum bet that happened to pay off over 2005 to 2018 and is not guaranteed to persist.
- **Risk parity is a cheap upgrade on 1/N:** similar return, lower risk, same turnover.

![Frontier](figures/efficient_frontier.png)

### VaR backtest, 2,821 days (28 violations expected)

| Model | Violations | Rate | Kupiec p-value | Independence p-value |
|---|---|---|---|---|
| Historical | 56 | 1.99% | 0.000 | 0.029 |
| Gaussian | 75 | 2.66% | 0.000 | 0.017 |
| Cornish-Fisher | **41** | **1.45%** | 0.023 | 0.023 |
| Monte Carlo (Student-t) | 52 | 1.84% | 0.000 | 0.017 |
| EWMA (RiskMetrics) | 60 | 2.13% | 0.000 | **0.182** |

- **Every model underestimates risk at 99%**: none passes the Kupiec test at the 1% level, the
  Gaussian one is the worst (fat tails ignored).
- Models on a long fixed window react too slowly: their violations **cluster** in the crisis
  (historical VaR: 32 of its 56 breaches fall in 2007 and 2008; independence rejected). Only **EWMA** removes the clustering (6 breaches in 2008 instead of 24), because its
  volatility updates within days, but its Gaussian tail still produces too many breaches.
- Takeaway: a good 99% VaR needs **both** a fast volatility model and a fat-tailed distribution,
  which motivates GARCH-t models (see
  [garch-volatility-forecasting](https://github.com/KlausKUEKOU/garch-volatility-forecasting)).

![VaR](figures/var_backtest.png)

## Limits

Survivorship bias (the 20 stocks are a 2018 selection), no dividends in prices, risk-free rate set to
zero. These favour the return-seeking strategies, so the risk conclusions are the robust part.

## Run it

```bash
pip install -r requirements.txt
python run_analysis.py      # about 3 minutes; writes results/ and figures/
```

```
data/stock_prices.csv     daily prices of 20 US stocks, 1990 to 2018
src/optimizers.py         portfolio rules, Ledoit-Wolf covariance, efficient frontier
src/risk.py               VaR models, Expected Shortfall, Kupiec and Christoffersen tests
run_analysis.py           walk-forward backtest, VaR backtest, charts
results/                  metrics tables (CSV)
figures/                  charts
```

Data: price file distributed with the open-source
[PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) test suite (MIT licence).

**Author:** Klaus KUEKOU
