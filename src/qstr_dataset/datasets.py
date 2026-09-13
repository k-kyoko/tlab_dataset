"""データセットを名前で引くための統一インターフェース。

生データの読み方（列の持ち方、単語ラベル、被験者番号の扱い）はデータセットごとに
違うので、その差をここで吸収し、どのデータセットも「非類似度行列を返すもの」
として扱えるようにする。
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from qstr_diversity import core

from . import paths

#: Amy データセットの刺激語（色10語 + 感情語13語）。行列の行・列ラベルになる。
AMY_WORDS: list[str] = [
    "red", "orange", "yellow", "green", "blue", "purple",
    "pink", "brown", "grey", "black",
    "happiness", "joy", "confidence", "calm", "boredom", "confusion",
    "anxiety", "fear", "sadness", "defeated", "anger", "envy", "disgust",
]

#: Amy データセットの被験者数。
AMY_N_SUB: int = 120


# --------------------------------------------------------------------------
# Amy
# --------------------------------------------------------------------------

def load_amy_raw(original: bool = False) -> pd.DataFrame:
    """Amy 形式の生 CSV を読む。先頭列はラベル列なので落とす。"""
    name = "Amy_dissimilarity_original.csv" if original else "Amy_dissimilarity.csv"
    raw = pd.read_csv(paths.source("Amy_pilot", "raw", name), header=None)
    return raw.drop(columns=0)


def load_amy_bdi() -> pd.DataFrame:
    """BDI スコア（Sub, BDI）を読む。"""
    return pd.read_csv(paths.source("Amy_pilot", "raw", "BDI_score.csv"))


def amy_dissimilarity(sub_no: int = -1) -> pd.DataFrame:
    """Amy データセットの非類似度行列。``sub_no=-1`` で全被験者平均。"""
    return core.create_dissim_amy(
        load_amy_raw(), sub_no=sub_no, unique_words=AMY_WORDS
    )


def amy_all_subjects() -> dict[int, pd.DataFrame]:
    """被験者番号 -> 非類似度行列。"""
    raw = load_amy_raw()
    return {
        sub: core.create_dissim_amy(raw, sub_no=sub, unique_words=AMY_WORDS)
        for sub in range(AMY_N_SUB)
    }


# --------------------------------------------------------------------------
# Angus
# --------------------------------------------------------------------------

AngusKind = Literal["simi", "pref"]


def load_angus_raw(kind: AngusKind = "simi") -> pd.DataFrame:
    """Angus 形式の生 CSV（列は col1, col2, pID, dist）を読む。"""
    name = {"simi": "simi_dists.csv", "pref": "pref_dists.csv"}[kind]
    return pd.read_csv(paths.source("Angus", name))


def angus_dissimilarity(sub_no: int = -1, kind: AngusKind = "simi") -> pd.DataFrame:
    """Angus データセットの非類似度行列。``sub_no=-1`` で全被験者平均。"""
    return core.create_dissim_angus(load_angus_raw(kind), sub_no=sub_no)


# --------------------------------------------------------------------------
# レジストリ
# --------------------------------------------------------------------------

DATASETS = {
    "amy": amy_dissimilarity,
    "angus": angus_dissimilarity,
}


def dissimilarity(name: str, **kwargs) -> pd.DataFrame:
    """データセット名から非類似度行列を得る。"""
    try:
        loader = DATASETS[name]
    except KeyError:
        raise KeyError(
            f"未知のデータセット: {name!r}（使えるのは {sorted(DATASETS)}）"
        ) from None
    return loader(**kwargs)
