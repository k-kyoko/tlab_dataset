"""プロジェクト内のパスを1か所で決める。

ノートブックに絶対パス（かつての ``/home/jovyan/work``）を直書きしないための入口。
どのディレクトリから起動しても同じ場所を指す。
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"pyproject.toml が見つかりません（探索開始: {start}）")


PROJECT_ROOT: Path = _find_project_root(Path(__file__).resolve())

#: 不変の生データ。既定では data/source（~/ResearchData_keep への symlink）。
#: 別マシンでは環境変数 QSTR_DATA_SOURCE で上書きする。
DATA_SOURCE: Path = Path(
    os.environ.get("QSTR_DATA_SOURCE", PROJECT_ROOT / "data" / "source")
)

#: 再生成できる中間生成物（h5 など）。Git にも Drive にも置かない。
DATA_INTERIM: Path = PROJECT_ROOT / "data" / "interim"

RESULTS_TABLES: Path = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES: Path = PROJECT_ROOT / "results" / "figures"


def source(*parts: str) -> Path:
    """生データのパスを返す。無い場合は理由のわかるエラーを出す。"""
    if not DATA_SOURCE.exists():
        raise FileNotFoundError(
            f"生データの置き場が見つかりません: {DATA_SOURCE}\n"
            "data/source は ~/ResearchData_keep への symlink です。"
            "別のマシンでは環境変数 QSTR_DATA_SOURCE に実際の場所を指定してください。"
        )
    path = DATA_SOURCE.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"ファイルがありません: {path}")
    return path


def interim(*parts: str) -> Path:
    """中間生成物のパスを返す。親ディレクトリは自動で作る。"""
    path = DATA_INTERIM.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def table(*parts: str) -> Path:
    """結果の表のパスを返す。親ディレクトリは自動で作る。"""
    path = RESULTS_TABLES.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def figure(*parts: str) -> Path:
    """結果の図のパスを返す。親ディレクトリは自動で作る。"""
    path = RESULTS_FIGURES.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
