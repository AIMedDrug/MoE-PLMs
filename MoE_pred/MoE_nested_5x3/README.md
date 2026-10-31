# MoE：5×3 嵌套交叉验证与路由统计（精简版）

入口为 `MoE_3PLMs.py`。代码直接取自 `MoE_nested_5x3_routing`，保留原有训练逻辑和默认参数。
本目录原有 `main_predict.py` 为旧入口；使用本版本请运行下面的新入口。

## 文件

- `MoE_3PLMs.py`：外层5折、内层3折选参、最终重训和外部测试。
- `moe_core.py`：模型、特征读取、归一化、训练与预测。
- `report_metrics.py`：pooled、逐蛋白质和macro指标。
- `routing_reports.py`：专家使用频率、路由熵、门控权重和样本分配统计及绘图。
- `search_space.py` / `search_space.json`：搜索空间读取校验与配置。
- `requirements.txt`：运行依赖。

以上5个Python文件必须放在同一目录，程序运行时也会读取它们以记录源代码哈希。
不包含测试、合成演示、缓存、历史文档、数据、模型权重或本机专用启动脚本。

## 安装与运行

在项目根目录 `MoE-PLMs-main` 下，激活安装了对应CPU/CUDA版本PyTorch的环境：

```powershell
python -m pip install -r MoE_pred/requirements.txt
```

准备三种PLM的突变减野生型特征，将下列示例中的 `YOUR_FEATURES` 替换为实际路径。
x1/x2/x3分别为ProstT5、ESMC-600M、ESM2-650M，训练和测试的分支顺序须一致。

```powershell
python MoE_pred/MoE_3PLMs.py `
  --train_x1 YOUR_FEATURES/S8754/ProstT5/result.pt `
  --train_x2 YOUR_FEATURES/S8754/esmc_600m/result.pt `
  --train_x3 YOUR_FEATURES/S8754/esm2_650m/new.pkl `
  --train_meta 'Data/ΔΔG/S8754.csv' `
  --test_x1 YOUR_FEATURES/S669/ProstT5/result.pt `
  --test_x2 YOUR_FEATURES/S669/esmc_600m-result.pt `
  --test_x3 YOUR_FEATURES/S669/esm2-650m/new.pkl `
  --test_meta 'Data/ΔΔG/S669.csv' `
  --branch-names ProstT5 ESMC-600M ESM2-650M `
  -o results/nested_5x3
```

元数据须有唯一、非空的 `ID` 和有限数值 `target` 列。
特征支持 `.pt` / `.pkl` 的ID到向量字典，或含ID列的数值特征CSV；特征须覆盖元数据中的ID。
默认从ID提取accession分组；自定义ID可指定 `--group-col` 和 `--protein-col`。
外部测试的四个参数可一起省略。输出目录须为空或不存在。

默认4个专家、Top-1，采用全部专家softmax后取Top-k，保留原概率、不重新归一化。
搜索配置见 `search_space.json`。默认初筛12候选、复核前3个、3个复核种子、最多100 epoch，完整流程最多约384次训练。
早停和选参使用Spearman/Pearson各占0.5的相关得分，S669不参与选参。

主要输出为 `final_model.pt`、外层/OOF指标、外部测试指标，以及各评估目录下的 `routing/` CSV和PNG/PDF。
`--save-outer-models` 可另存五个外层模型；`--no-routing-plots` 只输出路由CSV。
运行 `python MoE_pred/MoE_3PLMs.py --help` 可查看全部参数。
