# qstr_dataset

データ・適用コード・結果を束ねるプロジェクト。
同じ量（一般化マグニチュード）を2つの実装で計算し、結果を突き合わせる。

| backend | 実体 | 何をするか |
| --- | --- | --- |
| numeric | [qstr_diversity](../qstr_diversity) | t のグリッド上で genmag / spread を倍精度で数値計算。速い |
| exact | [genmag_exact](../genmag_exact) | t→∞ の極限を sympy で厳密計算。任意精度の有限 t 評価もできる |

## セットアップ

```bash
cd ~/Develop/qstr_dataset
uv sync
uv run jupyter lab
```

2つのライブラリは**兄弟ディレクトリを editable 参照**している（`pyproject.toml` の
`[tool.uv.sources]`）。ライブラリ側を編集すると即座に反映される。
そのため次の3つは兄弟に置き続けること。

```
~/Develop/
├── qstr_diversity/
├── genmag_exact/
└── qstr_dataset/
```

## ディレクトリ

```
src/qstr_dataset/
├── paths.py      プロジェクト内のパスを1か所で決める（絶対パスを直書きしない）
├── datasets.py   "amy" / "angus" → 非類似度行列 を返す統一インターフェース
├── backends.py   numeric / exact / highprec を同じ呼び方で叩く薄いラッパ
└── compare.py    2 backend の突き合わせ

notebooks/        探索用（出力はコミットしない）
data/
├── source        → ~/ResearchData_keep への symlink。不変の生データ。Git 外
└── interim/      再生成できる中間生成物（h5 など）。Git 外
results/
├── tables/       結果の表（csv）。軽いのでコミットする
└── figures/      図。再生成できるので Git 外
```

## データの置き場

生データの正本は `~/ResearchData_keep`（Google Drive でバックアップ）。
`data/source` はそこへの symlink で、Git では追跡していない。

別のマシンで動かすときは環境変数で場所を指定する。

```bash
export QSTR_DATA_SOURCE=/path/to/ResearchData_keep
```

## 使い方

```python
from qstr_dataset import datasets, backends, compare

# 全被験者平均の非類似度行列
dissim = datasets.dissimilarity("amy")

# 数値 backend：t のグリッド全体
curve = backends.numeric_curve(dissim)

# 厳密 backend：t→∞ の極限
backends.exact_limit(dissim)

# 2 backend の突き合わせ（数値 plateau が整数から外れた被験者だけ）
table = compare.compare_limits(datasets.amy_all_subjects(), only_mismatched=True)
```

## 2 backend の使い分け

数値 backend は t を大きくすると genmag が整数（＝有効な点の個数）に収束する。
収束していれば厳密版と一致するので、厳密計算を回す必要はない。

問題は、倍精度の擬似逆行列が効かなくなって**非整数の plateau に張り付く**被験者で、
そこだけを厳密 backend で計算し直す。`compare_limits` の `only_mismatched=True`（既定）は
この絞り込みをする。

厳密計算は 23×23 で数分以上かかるので、全員に回すのは現実的ではない。

```python
# 例：Amy データセットで数値 plateau が整数から外れた被験者を洗い出して厳密計算
table = compare.compare_limits(datasets.amy_all_subjects())
table.to_csv(paths.table("amy_numeric_vs_exact.csv"))
```
