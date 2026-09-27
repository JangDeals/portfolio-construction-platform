"""
Visualization module.
Reusable plotting functions for market data exploration, risk, and
performance analysis. All plots are saved to figures/ and returned for
inline notebook display.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.utils.paths import FIGURES_DIR

# Consistent styling across all plots in the project
plt.style.use("seaborn-v0_8-darkgrid")
FIGURE_SIZE = (12, 6)


def plot_price_evolution(prices: pd.DataFrame, filename: str = "price_evolution.png"):
    """
    Plot historical price evolution for all assets.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with dates as index, tickers as columns.
    filename : str
        Filename to save the figure as, inside FIGURES_DIR.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for ticker in prices.columns:
        ax.plot(prices.index, prices[ticker], label=ticker, linewidth=1)

    ax.set_title("Asset Price Evolution (2008–Present)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="upper left", ncol=3, fontsize=9)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")

    return fig

# --- Return Distribution Plotting Function ---
def plot_return_distributions(simple_returns: pd.DataFrame, filename: str = "return_distributions.png"):
    """
    Plot histograms of daily simple returns for all assets, arranged in a grid.
    """
    n_assets = len(simple_returns.columns)
    n_cols = 3
    n_rows = int(np.ceil(n_assets / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, ticker in enumerate(simple_returns.columns):
        axes[i].hist(simple_returns[ticker], bins=100, color="steelblue", edgecolor="black", alpha=0.7)
        axes[i].set_title(f"{ticker} Daily Return Distribution")
        axes[i].set_xlabel("Daily Return")
        axes[i].set_ylabel("Frequency")

    # Hide any unused subplot slots (if n_assets doesn't divide evenly into the grid)
    for j in range(n_assets, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")

    return fig

# --- Correlation Heatmap Plotting Function ---
def plot_correlation_heatmap(simple_returns: pd.DataFrame, filename: str = "correlation_heatmap.png"):
    """
    Plot a correlation heatmap of daily simple returns across all assets.
    """
    corr_matrix = simple_returns.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_matrix.columns)

    # Annotate each cell with its correlation value
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                     ha="center", va="center", color="black", fontsize=8)

    ax.set_title("Asset Correlation Matrix (Daily Returns)")
    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")

    return fig

# --- Rolling Volatility Plotting Function ---
def plot_rolling_volatility(simple_returns: pd.DataFrame, window: int = 60, filename: str = "rolling_volatility.png"):
    """
    Plot annualized rolling volatility for all assets using the given window
    (in trading days).
    """
    rolling_vol = simple_returns.rolling(window=window).std() * np.sqrt(252)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for ticker in rolling_vol.columns:
        ax.plot(rolling_vol.index, rolling_vol[ticker], label=ticker, linewidth=1)

    ax.set_title(f"{window}-Day Rolling Annualized Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility")
    ax.legend(loc="upper left", ncol=3, fontsize=9)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")

    return fig

# --- Rolling Correlation Plotting Function ---
def plot_rolling_correlation(simple_returns: pd.DataFrame, benchmark: str = "SPY", window: int = 60, filename: str = "rolling_correlation.png"):
    """
    Plot rolling correlation of each asset against a benchmark (default SPY).
    """
    other_assets = [col for col in simple_returns.columns if col != benchmark]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for ticker in other_assets:
        rolling_corr = simple_returns[ticker].rolling(window=window).corr(simple_returns[benchmark])
        ax.plot(rolling_corr.index, rolling_corr, label=f"{ticker} vs {benchmark}", linewidth=1)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"{window}-Day Rolling Correlation vs {benchmark}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    ax.legend(loc="upper left", ncol=3, fontsize=9)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")

    return fig

# --- Cumulative Performance Plotting Function ---
def plot_cumulative_performance(cumulative_returns: pd.DataFrame, filename: str = "cumulative_performance.png"):
    """
    Plot cumulative returns for all assets on one chart for direct comparison.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for ticker in cumulative_returns.columns:
        ax.plot(cumulative_returns.index, cumulative_returns[ticker] * 100, label=ticker, linewidth=1)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Cumulative Return by Asset (2008–Present)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper left", ncol=3, fontsize=9)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")

    return fig

def plot_efficient_frontier(frontier: pd.DataFrame, min_var_point: tuple, max_sharpe_point: tuple, 
                              equal_weight_point: tuple, filename: str = "efficient_frontier.png"):
    """
    Plot the efficient frontier with key reference portfolios marked.

    Parameters
    ----------
    frontier : pd.DataFrame
        Output of build_efficient_frontier, with target_return and volatility columns.
    min_var_point, max_sharpe_point, equal_weight_point : tuple
        (volatility, return) pairs for each reference portfolio.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(frontier["volatility"], frontier["target_return"], 
            color="steelblue", linewidth=2, label="Efficient Frontier (swept)")

    ax.scatter(*min_var_point, color="green", s=150, marker="*", 
               label="Minimum Variance Portfolio", zorder=5)
    ax.scatter(*max_sharpe_point, color="red", s=150, marker="*", 
               label="Maximum Sharpe Portfolio", zorder=5)
    ax.scatter(*equal_weight_point, color="orange", s=100, marker="o", 
               label="Equal Weight Portfolio", zorder=5)

    ax.set_title("Efficient Frontier")
    ax.set_xlabel("Volatility (Annualized)")
    ax.set_ylabel("Expected Return (Annualized)")
    ax.legend(loc="upper left")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")

    return fig