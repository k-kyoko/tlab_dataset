"""2つの計算 backend を同じ呼び方で叩くための薄いラッパ。

- ``numeric``  : qstr_diversity。t のグリッド上で倍精度の数値計算。速い。
- ``exact``    : genmag_exact。t→∞ の極限を sympy で厳密に計算。
- ``highprec`` : genmag_exact。有限の t を任意精度で評価。

同じ量を別の方法で出しているので、突き合わせると数値計算がどこで崩れるかが見える。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from genmag_exact import magnitude_limit as ml
from qstr_diversity import core


def default_t_grid() -> np.ndarray:
    """これまでの解析で使ってきた t のグリッド（10^-4 〜 10^8、計 560 点）。

    10^-1 〜 10^2 を細かく取ってあるのは、この帯で genmag が動くため。
    """
    return np.concatenate([
        np.logspace(-4, -1, 20, endpoint=False),
        np.logspace(-1, 2, 500, endpoint=False),
        np.logspace(2, 4, 20, endpoint=False),
        np.logspace(4, 8, 20, endpoint=True),
    ])


def _as_matrix(dissim: pd.DataFrame | np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    if isinstance(dissim, pd.DataFrame):
        return dissim.to_numpy()
    return np.asarray(dissim)


# --------------------------------------------------------------------------
# numeric backend (qstr_diversity)
# --------------------------------------------------------------------------

def numeric_at(dissim: pd.DataFrame, t: float) -> dict[str, Any]:
    """1点の t での genmag / spread / 類似度行列の正定値性。"""
    dist = core.cal_dissim2dist(dissim)
    sim = core.cal_dist2sim(dist, t)
    return {
        "t": float(t),
        "genmag": core.cal_sim2genmag(sim),
        "spread": core.cal_dist2spread(dist, t),
        "positive_definite": core.check_positivedefinite(sim.to_numpy()),
    }


def numeric_curve(
    dissim: pd.DataFrame, t_grid: Sequence[float] | None = None
) -> pd.DataFrame:
    """t のグリッド全体での曲線。index が t、列が genmag / spread / positive_definite。"""
    if t_grid is None:
        t_grid = default_t_grid()
    dist = core.cal_dissim2dist(dissim)
    rows = []
    for t in t_grid:
        sim = core.cal_dist2sim(dist, t)
        rows.append({
            "t": float(t),
            "genmag": core.cal_sim2genmag(sim),
            "spread": core.cal_dist2spread(dist, t),
            "positive_definite": core.check_positivedefinite(sim.to_numpy()),
        })
    return pd.DataFrame(rows).set_index("t")


# --------------------------------------------------------------------------
# exact backend (genmag_exact)
# --------------------------------------------------------------------------

def exact_limit(dissim: pd.DataFrame) -> dict[str, Any]:
    """t→∞ での genmag を厳密に求める。距離が整数・有理数であることが前提。"""
    result = ml.exact_limit(_as_matrix(dissim))
    return {
        "limit": result.limit,
        "limit_float": float(result.limit.evalf()),
        "rank_at_infinity": result.rank_at_infinity,
        "generic_rank": result.generic_rank,
        "method": result.method,
        "result": result,
    }


def highprec_at(dissim: pd.DataFrame, t: float, dps: int = 50) -> dict[str, Any]:
    """有限の t を任意精度で評価する。A(t) が非特異な t でのみ使える。"""
    value = ml.schur_value(_as_matrix(dissim), t, dps=dps)
    return {
        "t": float(value.t),
        "total": float(value.total),
        "range_part": float(value.range_part),
        "kernel_correction": float(value.kernel_correction),
        "decimal_precision": value.decimal_precision,
        "value": value,
    }
