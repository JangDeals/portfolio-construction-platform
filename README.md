# Multi-Asset Portfolio Construction, Optimization & Risk Analytics Platform

A quantitative investment research platform for constructing, optimizing, testing,
and monitoring a diversified multi-asset portfolio — built end-to-end in Python,
from raw market data acquisition through an interactive recommendation app.

> **For educational and research purposes only. Nothing in this repository, its
> outputs, or its live application constitutes financial advice.**

---

## Overview

This project builds a small, working version of what a quantitative portfolio
management system actually does: it acquires and validates market data, measures
risk and performance, constructs portfolios under several competing philosophies,
optimizes allocations subject to real-world constraints, backtests those
strategies honestly (including out-of-sample), stress-tests them against real
and hypothetical crises, and finally serves a recommendation through an
interactive application.

The guiding principle throughout is **empirical honesty over a polished
narrative**: every claim in this repository is backed by a specific, reproducible
test, and where a result was disappointing, inconclusive, or contradicted an
earlier finding, that result is reported rather than hidden.

### The business problem

Investors need a repeatable, defensible process for building and maintaining a
diversified portfolio — not ad hoc allocation decisions. This project asks: given
a fixed, economically diverse asset universe, how should capital be allocated to
balance return and risk, and how should that allocation be maintained over time
as markets move and evidence accumulates?

---

## Asset Universe

Nine ETFs, deliberately chosen so each plays a distinct economic role, spanning
four asset classes:

| Ticker | Role |
|---|---|
| SPY | US large-cap equity |
| QQQ | US growth / tech-heavy equity |
| VEA | Developed international equity |
| VWO | Emerging market equity |
| IEF | Intermediate-term US Treasuries |
| TLT | Long-term US Treasuries |
| SHY | Short-term Treasuries / cash proxy |
| VNQ | US real estate (REITs) |
| GLD | Gold |

Historical data spans **2008-01-02 to present**, daily frequency. The start date
is not arbitrary: VEA's actual inception (July 2007) is the latest inception date
among all nine assets, so 2008 is the earliest point at which every asset in the
universe has genuine, simultaneous trading history — a deliberate choice to avoid
mixing assets with unequal history lengths in the same covariance matrix.

---

## System Architecture

Market Data (yfinance)
│
Data Validation & Quality Checks
│
Return Engine (simple, log, cumulative, annualized)
│
Portfolio Analytics & Risk Engine
│
Portfolio Construction (7 competing methods)
│
Constrained Optimization
│
Backtesting (single-split AND walk-forward)
│
Stress Testing & Factor Analysis
│
ML Volatility Forecasting
│
Recommendation Engine
│
Streamlit Application


Each stage reads from the stage before it and produces a clean, independently
reusable output — the same functions used in research notebooks are imported
directly by the Streamlit app, with no duplicated logic.

---

## Data Pipeline

Raw price and volume data is downloaded via `yfinance`, validated, and saved
locally to `data/raw/` — but **this raw data is not committed to version
control** (see `.gitignore`). It is fully regenerable from code, and storing it
in Git would bloat the repository with a large, disposable artifact.

**To regenerate the data**, from the project root with the environment active:

```python
from src.data.loader import download_asset_data, save_raw_data
from src.config import TICKERS, START_DATE, END_DATE

prices, volumes = download_asset_data(TICKERS, START_DATE, END_DATE)
save_raw_data(prices, volumes, TICKERS, START_DATE, END_DATE)
```

This must be run once before any notebook or the Streamlit app can function,
since both expect `data/raw/asset_prices.csv` to already exist.

Every download also produces `asset_prices_metadata.json`, recording exactly
when the data was pulled, the requested vs. actual date range, and row/ticker
counts — so any result can be traced back to the exact data snapshot that
produced it.

### Data validation

Six independent checks run against every download: dimension/ticker
completeness, duplicate dates, missing values, date-continuity gaps, suspicious
price values (zero/negative prices, extreme single-day moves), and volume
anomalies. One genuine issue was found and fixed during development: yfinance
occasionally returns an incomplete row for the most recent trading day (or lags
a subset of tickers by a day or two), which is now automatically dropped rather
than silently corrupting downstream calculations.

---

## Methodology

### Portfolio construction methods

Seven distinct construction philosophies are implemented, each making a
different tradeoff between simplicity, reliance on estimated data, and investor
judgment:

| Method | Uses covariance? | Uses expected return? | Note |
|---|---|---|---|
| Equal Weight | No | No | Zero estimation error; ignores all risk/return data |
| Custom Allocation | No | No | Encodes strategic judgment; inherently subjective |
| Inverse Volatility | Standalone vol only | No | Ignores correlation entirely |
| Minimum Variance | Yes | No | Can produce extreme, single-asset concentration if unconstrained |
| Maximum Sharpe | Yes | Yes | Most sensitive to return-estimation error |
| Risk Parity | Yes | No | Equalizes each asset's marginal risk contribution |
| Maximum Diversification | Yes | No | Maximizes diversification benefit relative to standalone risk |

**Unconstrained Minimum Variance and Maximum Sharpe were found to produce
extreme, unrealistic concentration** (roughly 98% in a single low-volatility
asset, and a 3-asset concentration, respectively) — this is a well-documented,
real limitation of naive Markowitz optimization, not an implementation defect.
Practical use requires constraints.

### Constraints

Constrained optimization supports: maximum weight per asset, a uniform minimum
weight floor, asset-class bounds (e.g., equity between 40–70%), and a
concentration cap via the Herfindahl–Hirschman Index (HHI). Asset-class bounds
alone were found **not** to guarantee genuine diversification — an optimizer can
satisfy class-level bounds while concentrating on a single security within a
class. An HHI cap was needed to force genuine security-level spread.

### Risk & performance framework

Volatility, downside volatility, maximum drawdown, Historical and Parametric
Value-at-Risk, Conditional VaR (Expected Shortfall), tracking error, beta, and
concentration (HHI) on the risk side; CAGR, Sharpe, Sortino, Calmar, and
Information Ratio on the performance side. Parametric CVaR is deliberately not
implemented — it inherits the same normal-distribution assumption as Parametric
VaR, applied even deeper into the tail, precisely where real asset returns
(especially real estate and emerging markets) were found to deviate most from
normality.

### Backtesting

Two distinct methodologies are implemented and clearly distinguished:

- **Single train/test split**: weights estimated once on an earlier period, held
  fixed, and evaluated on a later, unseen period.
- **Walk-forward validation**: weights re-estimated annually on an expanding
  window, tested only on the following, unseen year, then stitched into one
  continuous out-of-sample return series. This is the more realistic test, since
  it mirrors how a real strategy would actually be operated.

All backtests train on chronologically earlier data and test on chronologically
later data — no random splitting is used anywhere in this project, since a
random split for time-series data risks training on information from the future.

### Stress testing

Historical scenarios (2008 Global Financial Crisis, 2020 COVID crash, 2022
rate-hiking bear market) are evaluated using real, dated returns. Hypothetical
scenarios (an equity-only crash, an inflation/rate shock, and a simultaneous
equity-and-bond crash) apply assumed instantaneous shocks. These two categories
are never blended or presented interchangeably.

---

## Key Results

These are reported as found, including results that were unflattering or
contradicted earlier findings.

1. **Diversification is real and measurable.** A simple equal-weighted 9-asset
   portfolio reduced volatility by roughly 33% relative to a naive weighted
   average of the individual assets' own volatilities — a direct, computed proof
   of the covariance-driven mechanism behind Modern Portfolio Theory, not just a
   textbook claim.

2. **Unconstrained optimization is not investable.** Minimum Variance
   concentrated ~98% in a single asset; Maximum Sharpe concentrated in 3 of 9
   assets, zeroing out core holdings like broad US equity entirely. Realistic
   constraints (asset-class bounds, an HHI cap) were required to produce
   portfolios a real investment committee could accept.

3. **A naive 60/40 benchmark was hard to beat.** Judged on the full historical
   period, only one of six constructed strategies (Maximum Sharpe) clearly
   outperformed a simple 60% equity / 40% bond benchmark on a risk-adjusted
   basis — a legitimate, if humbling, finding about the value of added
   complexity.

4. **In-sample outperformance did not survive a single frozen out-of-sample
   test, but did survive walk-forward re-optimization.** A Maximum Sharpe
   portfolio built once on early data and held unchanged underperformed simple
   Equal Weighting on later, unseen data. The same *method*, re-optimized
   annually rather than frozen, performed competitively. This distinguishes a
   weakness in a static estimate from a weakness in a method.

5. **No portfolio is universally "safe."** Minimum Variance — the strongest
   performer in the 2008 and 2020 crises — was the *worst* performer under a
   simulated inflation/rate shock, because its entire risk-reduction strategy
   depends on the exact asset class (bonds) that this type of shock damages
   most. Safety is conditional on which economic force is driving the crisis.

6. **A more complex ML model performed worse, consistently.** For volatility
   forecasting, Linear Regression outperformed Random Forest, which outperformed
   XGBoost — accuracy degraded monotonically as model complexity increased. A
   more accurate volatility forecast also did not translate into a better
   risk-adjusted portfolio outcome when applied inside an Inverse Volatility
   construction method.

7. **Portfolio concentration predicts drift speed, but not by itself.** A highly
   concentrated portfolio (Maximum Sharpe) breached a 5-percentage-point drift
   threshold in under a year; a diversified one (Equal Weight) took over twice
   as long. But concentration in a *low-volatility* asset (Minimum Variance)
   drifts far more slowly than concentration alone would predict — drift speed
   depends on both concentration and the volatility of what is concentrated.

8. **Walk-forward testing of the actual, deployed risk profiles** (not generic
   unconstrained strategies) showed the Moderate and Aggressive profiles
   outperforming the 60/40 benchmark on a risk-adjusted, out-of-sample basis.
   This is the strongest evidence in the project, though it still reflects one
   realized historical period.

---

## Limitations

- **Estimation risk.** Every optimization here uses historical average returns
  as a proxy for expected future returns — a well-documented, unreliable
  estimator. Maximum Sharpe optimization is especially sensitive to this.
- **Not a pristine holdout.** The risk-profile constraint templates (asset-class
  bounds, weight caps) were designed with knowledge of results across the full
  historical record. Even the walk-forward validation is not a fully blind test
  of the constraint design itself.
- **Simplified transaction costs.** Costs are modeled as a flat rate per unit of
  portfolio turnover. Bid-ask spreads, market impact, and taxes are not modeled.
- **Simplified factor analysis.** Only Market, Momentum, and Volatility factor
  exposures are covered. Value, Quality, and Size factors require external
  fundamental data not readily available for this ETF universe and are
  explicitly out of scope.
- **A single realized historical path.** All backtests, however rigorous their
  methodology, are still drawn from one specific stretch of market history.
  Results should be read as evidence, not as a guarantee.

---

## Installation

Requires Python 3.10–3.11 (managed via a dedicated Conda environment).

```bash
conda create -n portfolio_platform python=3.11 -y
conda activate portfolio_platform

git clone https://github.com/JangDeals/portfolio-construction-platform.git
cd portfolio-construction-platform

pip install -r requirements.txt
pip install -e .
```

The last step installs `src/` as an editable local package, so it can be
imported as `from src.data.loader import ...` from any notebook, script, or the
Streamlit app, without manual path configuration.

Then regenerate the market data as described in **Data Pipeline** above before
running anything else.

## Running the Application

```bash
streamlit run streamlit_app/app.py
```

Opens an interactive wizard: select a risk profile, review the recommended
portfolio, inspect its risk/performance metrics and position on the efficient
frontier, see out-of-sample walk-forward validation for that exact profile,
review a hypothetical rebalancing scenario, and download a full Markdown report
summarizing the entire recommendation.

## Running Tests

```bash
pytest tests/ -v
```

A suite of unit and edge-case tests covering return calculations, portfolio
theory, the risk engine, and optimization constraint compliance — each verified
against hand-computable expected results, not just checked for "runs without
error."

---

## Project Structure
Portfolio_Construction_Platform/
│
├── data/
│ ├── raw/ # Downloaded price/volume data (not in Git — see Data Pipeline)
│ ├── processed/ # Cleaned/derived data
│ └── external/ # Non-yfinance reference data
│
├── notebooks/ # Research notebooks, one per development stage
│
├── src/ # Installable package — all reusable production logic
│ ├── data/ # Acquisition and validation
│ ├── features/ # Return calculations
│ ├── portfolio/ # Construction, portfolio theory, benchmarks, factors, recommendation engine
│ ├── optimization/ # Mean-variance and advanced (risk parity, max diversification) optimizers
│ ├── risk/ # Risk engine (VaR, CVaR, drawdown, HHI, beta)
│ ├── performance/ # Performance ratios (Sharpe, Sortino, Calmar, IR)
│ ├── backtesting/ # Single-split and walk-forward backtests, rebalancing, stress testing
│ ├── forecasting/ # ML volatility forecasting
│ └── utils/ # Paths, visualization, model persistence, report generation, config
│
├── models/ # Saved trained models + metadata (models themselves not in Git)
├── reports/ # Generated analysis reports
├── figures/ # Saved charts
├── streamlit_app/ # Interactive application
├── tests/ # Automated test suite
│
├── pyproject.toml # Package configuration (enables pip install -e .)
├── requirements.txt
├── .gitignore
└── README.md


---

## Disclaimer

This project is a personal research and learning exercise. It is not
investment advice, and no output from this repository or its application
should be used to make real investment decisions.