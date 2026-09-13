"""2つの backend の結果を突き合わせる。

数値 backend は t を大きくすると genmag が整数に収束するはずだが、倍精度の
擬似逆行列が効かなくなる被験者では非整数の plateau に張り付く。
そこを厳密 backend で計算し直して差を見る。
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import backends

#: 数値 backend の plateau を見るときの既定の t（グリッドの最大値）。
PLATEAU_T: float = 1e8


def compare_limits(
    dissims: Mapping[int, pd.DataFrame],
    plateau_t: float = PLATEAU_T,
    only_mismatched: bool = True,
    atol: float = 1e-8,
) -> pd.DataFrame:
    """被験者ごとに「数値の plateau」と「厳密な極限」を並べた表を返す。

    Parameters
    ----------
    dissims
        被験者番号 -> 非類似度行列。
    plateau_t
        数値 backend を評価する t。
    only_mismatched
        True（既定）なら、数値 plateau が整数から外れている被験者だけを厳密計算する。
        厳密計算は 23x23 で数分以上かかるので、全員に回すのは現実的でない。
        数値 backend が整数に収束していれば厳密版と一致するため、
        食い違っている被験者だけを見れば足りる。
    atol
        整数とみなす許容誤差。
    """
    rows = []
    for sub, dissim in dissims.items():
        numeric = backends.numeric_at(dissim, plateau_t)
        plateau = numeric["genmag"]
        is_integer = bool(np.isclose(plateau, round(plateau), atol=atol))

        if only_mismatched and is_integer:
            continue

        exact = backends.exact_limit(dissim)
        rows.append({
            "subject": sub,
            "numeric_plateau": plateau,
            "exact_limit": exact["limit_float"],
            "exact_limit_symbolic": str(exact["limit"]),
            "difference": plateau - exact["limit_float"],
            "rank_at_infinity": exact["rank_at_infinity"],
            "numeric_plateau_is_integer": is_integer,
            "positive_definite_at_plateau_t": numeric["positive_definite"],
            "method": exact["method"],
        })

    columns = [
        "subject", "numeric_plateau", "exact_limit", "exact_limit_symbolic",
        "difference", "rank_at_infinity", "numeric_plateau_is_integer",
        "positive_definite_at_plateau_t", "method",
    ]
    return pd.DataFrame(rows, columns=columns).set_index("subject")
