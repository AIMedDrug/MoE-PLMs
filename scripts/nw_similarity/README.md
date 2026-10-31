# NW 跨数据集序列相似性

保留论文 A/B 图所需的代码、输入与结果。可单独上传整个目录，也可随仓库上传。

## 文件

- `NwEngine.cs`、`BLOSUM62.txt`：严格全局 Needleman–Wunsch 计算与替换矩阵。
- `unique_sequences.tsv`：四个数据集按完整 `wt_seq` 去重后的输入；无表头，三列依次为数据集、序列 UID、WT 序列。共 776 条，S8754/S669/S4346/S571 分别为 298/93/346/39 条。
- `S8754_vs_S669_similarity_pct.csv`、`S4346_vs_S571_similarity_pct.csv`：全部 41,208 对的相似性百分比；行是测试序列，列是训练序列，UID 对应 TSV。
- `plot_ab.py`、`NW_similarity_AB.png`：最终 A/B 图的绘图代码和 350 dpi 图片。
- `run.ps1`：重新计算两份矩阵并绘图的入口。
- `methods.json`：固定参数、输入校验值和原始运行环境版本。原始 CSV 路径相对于本目录。

## 复现

需要 Windows PowerShell/.NET、Python 3.8+、NumPy、matplotlib。原始运行版本见 `methods.json`。

```powershell
python -m pip install numpy matplotlib
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\nw_similarity\run.ps1
```

以上命令在仓库根目录执行。若单独上传本目录，在本目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1
```

仅重新绘图（跨平台，在本目录执行）：

```powershell
python plot_ab.py
```

计算使用随附的去重序列，不依赖旧机器上的绝对路径，也不需要 `all_pairs.csv`。输入来自仓库 `Data/ΔΔG` 与 `Data/ΔTm` 中四份原始 CSV 的 `wt_seq`，按首次出现顺序生成 UID。原始数据已在仓库中，故本目录不重复保存原始 CSV。

## 方法

严格全局 Needleman–Wunsch；BLOSUM62；线性缺口罚分 −5/残基，端部缺口同样计分。等分时依次优先对角线、测试序列缺口、训练序列缺口。Similarity = BLOSUM62 正分残基对数 / 全部比对列数 × 100%，分母包含缺口。图中每对唯一 WT 序列仅计一次。

A/B 图展示全部跨集序列对的分布，不能单独证明完全无同源或模型泛化能力。此处 Similarity 与相同残基比例 Identity 不同。
