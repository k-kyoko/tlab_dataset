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

## ノートブック

解析ノートブックはすべてここに集約している（ライブラリ側は関数だけを持つ）。

| ファイル | 出自 | 内容 |
| --- | --- | --- |
| `calc_allsubs.ipynb` | qstr_diversity | 全被験者の dissim → genmag / spread を計算して h5 に保存 |
| `calc_topbottom.ipynb` | qstr_diversity | BDI 上位／下位群での計算 |
| `visualize_mean.ipynb` | qstr_diversity | 平均行列の指標をプロット |
| `vizualize_allsubs.ipynb` | qstr_diversity | 全被験者・群ごとのプロット |
| `MDS_mean.ipynb` / `MDS_individual.ipynb` | qstr_diversity | MDS 埋め込みと可視化 |
| `stat_BDIandDI.ipynb` | qstr_diversity | BDI と多様性指標の相関 |
| `_paperfig.ipynb` | qstr_diversity | 論文用の図 |
| `genmag_exact_quickstart.ipynb` | genmag_exact | 厳密計算の動作確認 |
| `genmag_exact_calc_allsubs.ipynb` | genmag_exact | 全被験者の厳密計算（並列＋タイムアウト） |

パスはすべて `paths` 経由に書き換えてあり、`/home/jovyan/work` の直書きは残っていない。

```python
from qstr_dataset import paths

paths.source("Amy_pilot", "raw", "Amy_dissimilarity.csv")  # 生データ（symlink 先）
paths.interim("amy_processeddata_all.h5")                   # 中間生成物
paths.figure("MDS", "mean")                                 # 図の出力先
paths.table("genmag_exact_results.csv")                     # 表の出力先
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

## 2 backend の突き合わせ

**数値 backend の plateau が整数かどうかは、正しさの目安にならない。**
Amy データセットの 115 名（厳密計算が完走した分）:

| prev（数値 plateau） | 厳密値と一致 | 一致しない |
| --- | --- | --- |
| 整数 | 31 名 | **73 名** |
| 非整数 | 7 名 | 4 名 |

理由は擬似逆行列が rank の変わる点で不連続だから。つまり

```
lim_{t→∞} A(t)⁺  ≠  ( lim_{t→∞} A(t) )⁺
```

`np.linalg.pinv` は t が大きいところで小さい特異値を rcond で切り捨てる。これは
「極限をとってから擬似逆行列」を計算していることになり、rank に応じた整数に落ちる。
一方 `genmag_exact` が求めるのは「擬似逆行列の極限」で、消えていく方向の寄与が
有限に残るため値は一般に有理数になる（115 名中 35 名で厳密値そのものが非整数。
例: 63/8、81/5、181/13）。

整数に見えるのは pinv が rank に丸めた結果であって、収束の証拠ではない。
したがって**厳密計算は絞り込まずに全員に回す**。1 名あたり中央値 20 秒・最大 65 秒
なので、並列に回せば現実的な時間で終わる。

```python
from qstr_dataset import datasets, compare, paths

# 数値だけ先に見る（全員でも数秒）
compare.numeric_plateaus(datasets.amy_all_subjects())

# 全員を厳密計算して突き合わせる（並列・1名ずつタイムアウト付き）
table = compare.compare_limits(datasets.amy_all_subjects(), max_workers=8)
compare.summarize(table)          # 「整数か × 一致するか」のクロス集計
table.to_csv(paths.table("amy_numeric_vs_exact.csv"))
```

過去の実行結果は `results/tables/amy_numeric_vs_exact_20260829.csv` に置いてある
（2026-08-29 の 120 名分。ok 115 / timeout 4 / error 1）。
