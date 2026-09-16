"""データセットを名前で引くための統一インターフェース。

生データの読み方（列の持ち方、単語ラベル、被験者番号の扱い）はデータセットごとに
違うので、その差をここで吸収し、どのデータセットも「非類似度行列を返すもの」
として扱えるようにする。
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
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
# Togashi
# --------------------------------------------------------------------------

#: Togashi データの置き場（``data/source`` の下のフォルダ名）。
TOGASHI_FOLDER: str = "dissim_Togashi"


def togashi_subjects() -> list[str]:
    """被験者 ID の一覧。ファイル名 ``<id>_RDM.npy`` から拾う。"""
    folder = paths.source(TOGASHI_FOLDER)
    return sorted(path.name.removesuffix("_RDM.npy")
                  for path in folder.glob("*_RDM.npy"))


def togashi_dissimilarity(subject: str, zero_diagonal: bool = True) -> pd.DataFrame:
    """Togashi データセットの非類似度行列（93×93、整数）。

    ``zero_diagonal=True`` のとき、0 でない対角成分を 0 に直して警告を出す。
    生ファイルには自己比較が 1 と記録された成分があり（001 と 004 で 9 個、
    015 で 1 個）、そのままだと A(t) の対角が exp(-t) になって全ての値が変わる。
    行列ラベルは刺激名が無いので 0 始まりの整数。
    """
    matrix = np.load(paths.source(TOGASHI_FOLDER, f"{subject}_RDM.npy")).astype(float)
    nonzero_diagonal = int(np.count_nonzero(np.diag(matrix)))
    if nonzero_diagonal and zero_diagonal:
        warnings.warn(
            f"subject {subject}: {nonzero_diagonal} diagonal entries were not zero "
            "and have been set to zero",
            stacklevel=2,
        )
        np.fill_diagonal(matrix, 0.0)
    labels = list(range(len(matrix)))
    return pd.DataFrame(matrix, index=labels, columns=labels)


def togashi_all_subjects(zero_diagonal: bool = True) -> dict[str, pd.DataFrame]:
    """被験者 ID -> 非類似度行列。"""
    return {subject: togashi_dissimilarity(subject, zero_diagonal=zero_diagonal)
            for subject in togashi_subjects()}


# --------------------------------------------------------------------------
# レジストリ
# --------------------------------------------------------------------------

DATASETS = {
    "amy": amy_dissimilarity,
    "angus": angus_dissimilarity,
    "togashi": togashi_dissimilarity,
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
