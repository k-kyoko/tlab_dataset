"""2つの backend の結果を突き合わせる。

重要な事実（2026-09-13 に実データで確認）
-----------------------------------------
**数値 backend の plateau が整数かどうかは、正しさの目安にならない。**

Amy データセットの 115 名（厳密計算が完走した分）で:

======================  ================  ================
prev（数値 plateau）    厳密値と一致       一致しない
======================  ================  ================
整数                    31 名             **73 名**
非整数                   7 名               4 名
======================  ================  ================

理由は擬似逆行列が rank の変わる点で不連続だから。つまり

    lim_{t→∞} A(t)⁺  ≠  ( lim_{t→∞} A(t) )⁺

``np.linalg.pinv`` は t が大きいところで、小さい特異値を rcond で切り捨てる。
これは「極限をとってから擬似逆行列」を計算していることになり、rank に応じた
整数値にきれいに落ちる。一方 genmag_exact が求めるのは「擬似逆行列の極限」で、
消えていく方向の寄与が有限に残るため、値は一般に有理数になる
（115 名中 35 名で厳密値そのものが非整数。例: 63/8, 81/5, 181/13）。

整数に見えるのは pinv が rank に丸めた結果であって、収束の証拠ではない。
したがって**厳密計算は全員に回す**必要がある。1 名あたり中央値 20 秒・最大 65 秒
なので、並列に回せば現実的な時間で終わる。
"""

from __future__ import annotations

import multiprocessing as mp
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Mapping

import numpy as np
import pandas as pd
import sympy as sp
from genmag_exact import magnitude_limit as ml

from . import backends

#: 数値 backend の plateau を見るときの既定の t（既定グリッドの最大値）。
PLATEAU_T: float = 1e8

#: 厳密計算の 1 名あたりの上限（秒）。
DEFAULT_TIMEOUT_SEC: int = 300


# --------------------------------------------------------------------------
# 数値側（速い）
# --------------------------------------------------------------------------

def numeric_plateaus(
    dissims: Mapping[Any, pd.DataFrame], t: float = PLATEAU_T
) -> pd.DataFrame:
    """数値 backend だけを t=plateau_t で評価する。全員分でも数秒。"""
    rows = []
    for key, dissim in dissims.items():
        result = backends.numeric_at(dissim, t)
        rows.append({
            "subject": key,
            "numeric_plateau": result["genmag"],
            "spread": result["spread"],
            "positive_definite": result["positive_definite"],
        })
    return pd.DataFrame(rows).set_index("subject")


# --------------------------------------------------------------------------
# 厳密側（重いので並列＋タイムアウト）
# --------------------------------------------------------------------------

class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):  # pragma: no cover - シグナルハンドラ
    raise _Timeout


def _solve_one(key: Any, matrix: np.ndarray, timeout_sec: int) -> dict[str, Any]:
    """子プロセスで 1 件計算する。sympy の式は文字列にして返す（pickle 対策）。"""
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(timeout_sec)
    started = time.perf_counter()
    try:
        result = ml.exact_limit(matrix)
        return {
            "subject": key,
            "status": "ok",
            "exact_limit": str(result.limit),
            "rank_at_infinity": result.rank_at_infinity,
            "method": result.method,
            "sec": time.perf_counter() - started,
            "error": None,
        }
    except _Timeout:
        return {"subject": key, "status": "timeout", "exact_limit": None,
                "rank_at_infinity": None, "method": None,
                "sec": time.perf_counter() - started, "error": None}
    except Exception as exc:
        return {"subject": key, "status": "error", "exact_limit": None,
                "rank_at_infinity": None, "method": None,
                "sec": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}"}
    finally:
        signal.alarm(0)


def _to_float(expr: str | None) -> float | None:
    if expr is None:
        return None
    try:
        return float(sp.sympify(expr))
    except (TypeError, ValueError):
        return None  # sympy.oo など数値化できないもの


def exact_limits(
    dissims: Mapping[Any, pd.DataFrame],
    max_workers: int | None = None,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    progress: bool = True,
) -> pd.DataFrame:
    """厳密 backend を全件に回す。1 件ごとにタイムアウトを掛けて落ちても止めない。"""
    items = [(key, backends._as_matrix(d)) for key, d in dissims.items()]
    results: list[dict[str, Any]] = []
    started = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=mp.get_context("fork")
    ) as pool:
        futures = {
            pool.submit(_solve_one, key, matrix, timeout_sec): key
            for key, matrix in items
        }
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if progress:
                print(
                    f"\r[{done}/{len(items)}] subject={result['subject']} "
                    f"{result['status']} ({result['sec']:.1f}s)".ljust(70),
                    end="", flush=True,
                )
    if progress:
        print(f"\r完了: {len(items)} 件 / {time.perf_counter() - started:.1f}s".ljust(70))

    frame = pd.DataFrame(results)
    frame["exact_limit_float"] = frame["exact_limit"].map(_to_float)
    return frame.set_index("subject").sort_index()


# --------------------------------------------------------------------------
# 突き合わせ
# --------------------------------------------------------------------------

def compare_limits(
    dissims: Mapping[Any, pd.DataFrame],
    plateau_t: float = PLATEAU_T,
    max_workers: int | None = None,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    atol: float = 1e-6,
    progress: bool = True,
) -> pd.DataFrame:
    """「数値の plateau」と「厳密な極限」を並べた表を返す。

    絞り込みはしない。モジュール冒頭のとおり、数値 plateau が整数であることは
    厳密値と一致する根拠にならないため、全件を厳密計算する。
    """
    numeric = numeric_plateaus(dissims, plateau_t)
    exact = exact_limits(
        dissims, max_workers=max_workers, timeout_sec=timeout_sec, progress=progress
    )
    table = numeric.join(exact, how="outer")

    plateau = table["numeric_plateau"]
    limit = table["exact_limit_float"]
    table["difference"] = plateau - limit
    table["agrees"] = np.isclose(plateau, limit, atol=atol, equal_nan=False)
    table["numeric_plateau_is_integer"] = np.isclose(plateau, plateau.round())
    table["exact_limit_is_integer"] = np.isclose(limit, limit.round())

    columns = [
        "numeric_plateau", "exact_limit", "exact_limit_float",
        "difference", "agrees",
        "numeric_plateau_is_integer", "exact_limit_is_integer",
        "rank_at_infinity", "positive_definite", "status", "method", "sec", "error",
    ]
    return table[columns]


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """compare_limits の結果を「整数か × 一致するか」のクロス集計にする。"""
    ok = table[table["status"] == "ok"]
    return pd.crosstab(
        ok["numeric_plateau_is_integer"],
        ok["agrees"],
        rownames=["数値 plateau が整数"],
        colnames=["厳密値と一致"],
    )
