"""S669 inference only: restore weights and fitted scalers; never train."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr, weightedtau

class Sparse_MoE(nn.Module):
    def __init__(self, d_model, num_experts=4, top_k=1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.routing_mode = 'full_softmax_topk_no_renormalization'
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.d_model)
        
        # 筛选Top-K专家
        gate_logits = self.gate(x_flat)
        if self.routing_mode == 'topk_softmax_renormalized':
            top_k_values, top_k_indices = torch.topk(gate_logits, self.top_k, dim=1)
            gate_weights = torch.softmax(top_k_values, dim=1)
        else:
            gate_probabilities = torch.softmax(gate_logits, dim=1)
            gate_weights, top_k_indices = torch.topk(gate_probabilities, self.top_k, dim=1)
        # Keep the full-softmax probabilities; do not renormalize selected weights.
        # Diagnostic observer is installed only during held-out inference.
        # Capture the actual selected indices (including torch.topk tie decisions).
        observer = getattr(self, 'routing_observer', None)
        if observer is not None:
            observer(gate_logits.detach(), top_k_indices.detach(), gate_weights.detach())
        
        # 专家计算与加权
        expert_outputs = []
        for i in range(self.num_experts):
            mask = (top_k_indices == i).float()
            expert_out = self.experts[i](x_flat)
            weighted_out = expert_out.unsqueeze(1) * mask.unsqueeze(-1)
            expert_outputs.append(weighted_out)
        
        # 聚合输出
        moe_out = torch.sum(torch.stack(expert_outputs), dim=0)
        moe_out = torch.sum(moe_out * gate_weights.unsqueeze(-1), dim=1)
        return moe_out.view(batch_size, seq_len, self.d_model)


class MLPInteractionLayer(nn.Module):
    def __init__(self, d_model, hidden_dim=None, dropout=0.5):  # 微调dropout至0.4，平衡拟合与泛化
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim if hidden_dim else 2 * d_model  # 隐藏层维度默认设为输入的2倍
        
        # MLP结构：新增1层隐藏层，强化非线性交互
        self.mlp = nn.Sequential(
            # 第一层：输入拼接特征（3*d_model）→ 升维至hidden_dim
            nn.Linear(3 * d_model, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # 新增第二层隐藏层：强化非线性变换（核心修改点）
            nn.Linear(self.hidden_dim, self.hidden_dim),  # 新增隐藏层，维度保持hidden_dim
            nn.BatchNorm1d(self.hidden_dim),            # 批归一化稳定训练
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # 输出层：降维回3*d_model，拆分回三分支
            nn.Linear(self.hidden_dim, 3 * d_model),
            nn.LayerNorm(3 * d_model)
        )

    def forward(self, x1, x2, x3):
        # 保持与原版本一致的前向传播逻辑
        batch_size = x1.shape[0]
        x1_flat = x1.squeeze(1)  # [batch, d_model]
        x2_flat = x2.squeeze(1)
        x3_flat = x3.squeeze(1)
        combined = torch.cat([x1_flat, x2_flat, x3_flat], dim=1)  # [batch, 3*d_model]
        
        mlp_out = self.mlp(combined)  # 经过加深的MLP处理
        
        # 拆分回三分支
        x1_out = mlp_out[:, :self.d_model].unsqueeze(1)
        x2_out = mlp_out[:, self.d_model:2*self.d_model].unsqueeze(1)
        x3_out = mlp_out[:, 2*self.d_model:].unsqueeze(1)
        
        return x1_out, x2_out, x3_out


class CombinedMLPMoEModel(nn.Module):
    def __init__(self, input_dims, d_model=64, hidden_dim_mlp=None, num_experts=8, top_k=1, dropout=0.5):
        super().__init__()
        self.d_model = d_model
        
        # 三分支特征投影（统一维度）
        self.proj_x1 = nn.Linear(input_dims[0], d_model)
        self.proj_x2 = nn.Linear(input_dims[1], d_model)
        self.proj_x3 = nn.Linear(input_dims[2], d_model)
        
        # MLP特征交互层（替代原交叉注意力）
        self.mlp_interaction = MLPInteractionLayer(d_model, hidden_dim_mlp, dropout)
        
        # 稀疏MoE增强（保持不变）
        self.moe = Sparse_MoE(d_model, num_experts, top_k)
        
        # 特征融合与回归头（增加正则化）
        self.fusion_fc = nn.Linear(d_model * 3, d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,  # 可调整的注意力头数
            dropout=dropout,
            batch_first=True  # 确保batch维度在前
        )
        # 注意力输出后最终映射到1个输出值
        self.attention_output = nn.Linear(d_model, 1)

    def forward(self, x1, x2, x3):
        # 步骤1：特征投影+增加序列维度
        x1_proj = self.proj_x1(x1).unsqueeze(1)  # [batch, 1, d_model]
        x2_proj = self.proj_x2(x2).unsqueeze(1)
        x3_proj = self.proj_x3(x3).unsqueeze(1)
        
        # 步骤2：MLP特征交互（核心修改点）
        x1_mlp, x2_mlp, x3_mlp = self.mlp_interaction(x1_proj, x2_proj, x3_proj)
        
        # 步骤3：MoE增强+特征融合
        x1_moe = self.moe(x1_mlp).squeeze(1)  # [batch, d_model]
        x2_moe = self.moe(x2_mlp).squeeze(1)
        x3_moe = self.moe(x3_mlp).squeeze(1)
        
        fused = torch.cat([x1_moe, x2_moe, x3_moe], dim=1)  # [batch, 3*d_model]
        fused = self.fusion_fc(fused)
        fused = self.bn(fused)
        fused = self.dropout(fused)
        
        # 回归输出
        fused_seq = fused.unsqueeze(1)
        attn_output, _ = self.attention(
            query=fused_seq,
            key=fused_seq,
            value=fused_seq
        )  # [batch, 1, d_model]
        
        # 最终输出
        out = self.attention_output(attn_output.squeeze(1))  # [batch, 1]
        return out


def load_single_feat(feat_path, meta_df):
    suffix = Path(feat_path).suffix.lower()
    if suffix == '.csv':
        embed_df = pd.read_csv(feat_path, dtype={'ID': str})
    elif suffix in ('.pkl', '.pt'):
        embed = (pd.read_pickle(feat_path) if suffix == '.pkl' else
                 torch.load(feat_path, map_location='cpu', weights_only=True))
        if not isinstance(embed, dict):
            raise ValueError(f"{feat_path}: pkl/pt 文件必须是 ID 到特征向量的字典")
        vectors = {
            str(key): (value.detach().cpu().numpy() if torch.is_tensor(value)
                       else np.asarray(value)).reshape(-1)
            for key, value in embed.items()
        }
        embed_df = pd.DataFrame.from_dict(vectors, orient='index')
        embed_df.index.name = 'ID'
        embed_df = embed_df.reset_index()
    else:
        raise ValueError('仅支持 .pkl, .pt, .csv 格式')
    if 'ID' not in embed_df or embed_df['ID'].isna().any() or embed_df['ID'].duplicated().any():
        raise ValueError(f"{feat_path}: 特征 ID 必须存在、非空且唯一")
    # 左连接保留元数据顺序，缺失特征直接报错，禁止静默删除或错位。
    aligned = meta_df[['ID']].merge(embed_df, how='left', on='ID',
                                     sort=False, validate='many_to_one', indicator=True)
    if aligned['_merge'].ne('both').any():
        missing = aligned.loc[aligned['_merge'].ne('both'), 'ID'].head().tolist()
        raise ValueError(f"{feat_path}: 缺少样本特征: {missing}")
    features = aligned.drop(columns=['ID', '_merge']).apply(pd.to_numeric, errors='raise')
    if features.shape[1] == 0 or not np.isfinite(features.to_numpy(dtype=np.float32)).all():
        raise ValueError(f"{feat_path}: 特征必须为非空、有限数值")
    return features.astype(np.float32)


class CombinedRegHeadMoEModel(CombinedMLPMoEModel):
    """The original repeated-run model has a linear head and no attention."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.attention
        del self.attention_output
        self.reg_head = nn.Linear(self.d_model, 1)

    def forward(self, x1, x2, x3):
        branches = self.mlp_interaction(
            self.proj_x1(x1).unsqueeze(1), self.proj_x2(x2).unsqueeze(1),
            self.proj_x3(x3).unsqueeze(1))
        fused = torch.cat([self.moe(branch).squeeze(1) for branch in branches], dim=1)
        return self.reg_head(self.dropout(self.bn(self.fusion_fc(fused))))


def sha256(path):
    result = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def regression_metrics(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    nonconstant = len(y) > 1 and np.ptp(y) > 0 and np.ptp(pred) > 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return {
            'n_records': len(y),
            'mae': float(metrics.mean_absolute_error(y, pred)),
            'rmse': float(np.sqrt(metrics.mean_squared_error(y, pred))),
            'r2': float(metrics.r2_score(y, pred)) if len(y) > 1 and np.ptp(y) > 0 else np.nan,
            'pearson': float(pearsonr(y, pred)[0]) if nonconstant else np.nan,
            'spearman': float(spearmanr(y, pred)[0]) if nonconstant else np.nan,
            'kendall_weighted': float(weightedtau(y, pred)[0]) if nonconstant else np.nan,
        }


def run_prediction(data_root, output_dir=None, device='auto', batch_size=None):
    data_root = Path(data_root).expanduser().resolve()
    paths = {
        'checkpoint': data_root / 'weights/final_model.pt',
        'metadata': data_root / 'data/S669.csv',
        'ProstT5': data_root / 'data/S669_ProstT5.pt',
        'ESMC': data_root / 'data/S669_ESMC.pt',
        'ESM2': data_root / 'data/S669_ESM2.pkl',
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'{name}: {path}; 请先上传并解压 S669_inputs.zip，或修改 DATA_ROOT。')
    # If the bundled manifest exists, verify the weight/data identity before loading.
    manifest_path = data_root / 'input_manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for record in manifest['files']:
            path = data_root / record['file']
            if not path.is_file() or sha256(path) != record['sha256']:
                raise ValueError(f'文件完整性校验失败: {path}')

    package = torch.load(paths['checkpoint'], map_location='cpu', weights_only=True)
    required = {'state_dict', 'params', 'input_dims', 'feature_columns', 'scalers', 'routing_mode'}
    if not isinstance(package, dict) or not required.issubset(package):
        raise ValueError('需要包含超参数、scaler 和 routing_mode 的完整 final_model.pt 检查点。')
    if package['routing_mode'] not in ('full_softmax_topk_no_renormalization', 'topk_softmax_renormalized'):
        raise ValueError(f"不支持此检查点的路由方式: {package['routing_mode']}")
    if not all(len(package[key]) == 3 for key in ('input_dims', 'feature_columns', 'scalers')):
        raise ValueError('检查点必须包含三个特征分支。')
    params = package['params']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('当前环境未提供 CUDA，请改用 device="auto" 或 "cpu"。')
    batch_size = int(params['batch_size'] if batch_size is None else batch_size)
    if batch_size < 1:
        raise ValueError('batch_size 必须大于 0')
    print(f'Device: {device}; batch_size: {batch_size}')
    print(f'Checkpoint params: {params}')

    meta = pd.read_csv(paths['metadata'], dtype={'ID': str})
    if meta.empty or 'ID' not in meta or meta['ID'].isna().any() or meta['ID'].str.strip().eq('').any():
        raise ValueError('测试集必须非空，且包含非空 ID 列。')
    tensors = []
    for index, name in enumerate(('ProstT5', 'ESMC', 'ESM2')):
        # Left join by ID, retaining metadata row order and repeated measurements.
        features = load_single_feat(paths[name], meta)
        columns = package['feature_columns'][index]
        if list(features.columns) != columns:
            if set(features.columns) != set(columns):
                raise ValueError(f'{name}: 特征列与训练时不一致')
            features = features.loc[:, columns]
        if features.shape[1] != package['input_dims'][index]:
            raise ValueError(f'{name}: 特征维度不匹配')
        scaler = MinMaxScaler(feature_range=tuple(package.get('scaler_feature_range', [0, 1])))
        state = package['scalers'][index]
        for key in ('scale_', 'min_', 'data_min_', 'data_max_', 'data_range_'):
            value = state[key]
            value = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            if value.shape != (len(columns),) or not np.isfinite(value).all():
                raise ValueError(f'{name}: scaler 参数无效: {key}')
            setattr(scaler, key, value)
        scaler.n_features_in_ = len(columns)
        if all(isinstance(column, str) for column in columns):
            scaler.feature_names_in_ = np.asarray(columns, dtype=object)
        # Use exactly the training transform, with no test fitting or clipping.
        scaled = scaler.transform(features).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError(f'{name}: 归一化后出现非有限值')
        tensors.append(torch.from_numpy(scaled))
        print(f'{name}: {features.shape}')

    architecture = package.get('architecture')
    if architecture is None:
        architecture = 'mlp_moe_reg_head' if 'reg_head.weight' in package['state_dict'] else 'mlp_moe_attention'
    model_types = {'mlp_moe_reg_head': CombinedRegHeadMoEModel, 'mlp_moe_attention': CombinedMLPMoEModel}
    if architecture not in model_types:
        raise ValueError(f'不支持此检查点的模型结构: {architecture}')
    model = model_types[architecture](
        input_dims=package['input_dims'],
        **{key: params[key] for key in ('d_model', 'hidden_dim_mlp', 'num_experts', 'top_k', 'dropout')}
    ).to(device)
    model.moe.routing_mode = package['routing_mode']
    model.load_state_dict(package['state_dict'], strict=True)
    model.eval()
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size,
                        shuffle=False, drop_last=False, num_workers=0)
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            # reshape(-1) also handles a final batch containing only one sample.
            predictions.append(model(*(x.to(device) for x in batch)).cpu().numpy().reshape(-1))
    pred = np.concatenate(predictions)
    if len(pred) != len(meta) or not np.isfinite(pred).all():
        raise RuntimeError('预测数量不匹配或包含非有限值')

    output_dir = Path(output_dir) if output_dir is not None else data_root / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    result = meta[['ID']].copy()
    result['prediction'] = pred
    groups = meta['ID'].str.extract(r'^((?:rcsb_[A-Za-z0-9]{4}|uniprot_[A-Za-z0-9]+))_', expand=False)
    if groups.notna().all():
        result.insert(1, 'protein_id', groups.str.upper())
    scores = None
    if 'target' in meta:
        # Match original training/testing loader: targets were converted to float32.
        truth = pd.to_numeric(meta['target'], errors='raise').to_numpy(dtype=np.float32)
        if not np.isfinite(truth).all():
            raise ValueError('target 包含缺失值或非有限值')
        result['target'] = truth
        result['residual'] = pred - truth
        result['absolute_error'] = np.abs(pred - truth)
        scores = regression_metrics(truth, pred)
        pd.DataFrame([scores]).to_csv(output_dir / 'metrics.csv', index=False, encoding='utf-8-sig')
        if 'protein_id' in result:
            per_protein = []
            for protein, rows in result.groupby('protein_id', sort=True):
                per_protein.append({'protein_id': protein,
                                    **regression_metrics(rows['target'], rows['prediction'])})
            pd.DataFrame(per_protein).to_csv(output_dir / 'per_protein_metrics.csv', index=False, encoding='utf-8-sig')
        print(pd.DataFrame([scores]).to_string(index=False))
    result.to_csv(output_dir / 'predictions.csv', index=False, encoding='utf-8-sig')
    run_info = {
        'checkpoint': str(paths['checkpoint']), 'checkpoint_sha256': sha256(paths['checkpoint']),
        'inputs': {name: {'path': str(path), 'sha256': sha256(path)} for name, path in paths.items()},
        'n_records': len(meta), 'input_dims': package['input_dims'], 'params': params,
        'routing_mode': package['routing_mode'], 'architecture': architecture,
        'device': str(device), 'batch_size': batch_size,
        'python': platform.python_version(), 'torch': str(torch.__version__),
        'numpy': np.__version__, 'pandas': pd.__version__,
        'training_performed': False, 'scalers_fitted_on_test': False,
    }
    (output_dir / 'run_info.json').write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'完成 {len(result)} 条预测。结果目录: {output_dir.resolve()}')
    return result, scores, output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从 final_model.pt 恢复模型，对 S669 进行预测（不训练）')
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()
    run_prediction(args.data_root, args.output_dir, args.device, args.batch_size)
