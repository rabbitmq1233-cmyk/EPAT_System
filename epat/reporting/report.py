"""Text and (optional) HTML/PNG performance reports."""

from __future__ import annotations

import base64
import math
from pathlib import Path

import pandas as pd

from epat.engine.result import BacktestResult

_PCT_METRICS = {
    "total_return",
    "cagr",
    "annual_volatility",
    "max_drawdown",
    "hit_ratio",
}
_COUNT_METRICS = {"num_trades"}


def _fmt(key: str, value: float) -> str:
    if value is None:
        return "n/a"
    val = float(value)
    if math.isnan(val):
        return "nan"
    if math.isinf(val):
        return "inf"
    if key in _PCT_METRICS:
        return f"{val * 100:,.2f}%"
    if key in _COUNT_METRICS:
        return f"{int(val)}"
    if key == "final_equity":
        return f"{val:,.2f}"
    return f"{val:,.3f}"


def format_metrics(metrics: dict[str, float], title: str = "Performance") -> str:
    """Render a metrics dictionary as an aligned text block."""
    width = max((len(k) for k in metrics), default=10) + 2
    lines = [title, "-" * len(title)]
    for key, value in metrics.items():
        lines.append(f"{key:<{width}} {_fmt(key, value)}")
    return "\n".join(lines)


def _trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    if trades is None or trades.empty:
        return {"num_trades": 0.0, "avg_win": float("nan"), "avg_loss": float("nan")}
    pnl = trades["pnl"].dropna()
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    return {
        "num_trades": float(pnl.size),
        "avg_win": float(wins.mean()) if not wins.empty else float("nan"),
        "avg_loss": float(losses.mean()) if not losses.empty else float("nan"),
    }


def tearsheet(result) -> str:
    """Full text tearsheet for a backtest / portfolio / execution result."""
    name = getattr(result, "name", "result")
    blocks = [format_metrics(result.metrics, title=f"Performance - {name}")]
    stats = _trade_stats(getattr(result, "trades", None))
    blocks.append(
        format_metrics(
            {
                "num_trades": stats["num_trades"],
                "avg_win": stats["avg_win"],
                "avg_loss": stats["avg_loss"],
            },
            title="Trade statistics",
        )
    )
    if not result.equity.empty:
        blocks.append(
            f"Period: {result.equity.index[0].date()} -> {result.equity.index[-1].date()}"
        )
    return "\n\n".join(blocks)


def write_tearsheet(result: BacktestResult, path: str | Path) -> Path:
    """Write the text tearsheet to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tearsheet(result), encoding="utf-8")
    return path


def plot_equity(result: BacktestResult, path: str | Path | None = None):
    """Plot the equity curve with matplotlib if available.

    Returns the saved :class:`Path`, or ``None`` when matplotlib is not
    installed or no path is given.
    """
    try:  # pragma: no cover - environment dependent
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    if path is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    result.equity.plot(ax=ax, title=f"Equity - {getattr(result, 'name', 'result')}")
    ax.set_xlabel("date")
    ax.set_ylabel("equity")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _equity_png_base64(result) -> str | None:
    try:  # pragma: no cover - environment dependent
        import matplotlib

        matplotlib.use("Agg")
        import io

        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    result.equity.plot(ax=ax, title=f"Equity - {getattr(result, 'name', 'result')}")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def write_html_report(result: BacktestResult, path: str | Path) -> Path:
    """Write a self-contained HTML report (metrics + optional equity chart)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = "\n".join(
        f"<tr><td>{k}</td><td>{_fmt(k, v)}</td></tr>" for k, v in result.metrics.items()
    )
    img = _equity_png_base64(result)
    img_tag = (
        f'<img alt="equity curve" src="data:image/png;base64,{img}"/>' if img else ""
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>EPAT report - {getattr(result, 'name', 'result')}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1b1b1b; }}
h1 {{ font-size: 1.3rem; }} table {{ border-collapse: collapse; margin: 1rem 0; }}
td {{ border-bottom: 1px solid #eee; padding: 4px 12px; }}
img {{ max-width: 100%; height: auto; }}
</style></head>
<body>
<h1>Backtest report - {getattr(result, 'name', 'result')}</h1>
<table>{rows}</table>
{img_tag}
</body></html>
"""
    path.write_text(html, encoding="utf-8")
    return path
