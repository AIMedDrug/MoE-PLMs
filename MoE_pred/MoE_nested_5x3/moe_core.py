import argparse
import json
import random
from pathlib import Path
from sklearn.model_selection import GroupKFold
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from scipy.stats import spearmanr,pearsonr
from scipy.stats import weightedtau





# 1. 稀疏MoE模块（保持不变）
class Sparse_MoE(nn.Module):
    def __init__(self, d_model, num_experts=4, top_k=1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.d_model)
        
        # 筛选Top-K专家
        gate_logits = self.gate(x_flat)
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

# 2. 新增MLP特征交互层（核心修改：替代原交叉注意力层）
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

# 3. 主模型（MLP交互+MoE，核心修改）
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

# 4. 数据集类（保持不变）
class FeatureDataset(Dataset):
    def __init__(self, x1_feats, x2_feats, x3_feats, target):
        self.x1 = torch.tensor(x1_feats.values, dtype=torch.float32)
        self.x2 = torch.tensor(x2_feats.values, dtype=torch.float32)
        self.x3 = torch.tensor(x3_feats.values, dtype=torch.float32)
        self.target = torch.tensor(target.values, dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.x3[idx], self.target[idx]

# 5. 数据读取与蛋白质分组
# 使用来源和完整蛋白质 accession：rcsb_1A43 或 uniprot_P00004。
# 如需按序列或其他蛋白质标识分组，请显式指定 --group_col。
def protein_groups(meta_df, group_col=None):
    if group_col is not None:
        if group_col not in meta_df:
            raise ValueError(f"元数据缺少分组列: {group_col}")
        groups = meta_df[group_col]
    else:
        groups = meta_df['ID'].str.extract(r'^((?:rcsb_[A-Za-z0-9]{4}|uniprot_[A-Za-z0-9]+))_', expand=False)
        if groups.isna().any():
            raise ValueError("无法从所有 ID 提取 rcsb_PDB / uniprot_accession 标识，请指定 --group_col")
        groups = groups.str.upper()
    if groups.isna().any() or groups.astype(str).str.strip().eq('').any():
        raise ValueError("蛋白质分组标识不能缺失或为空")
    return groups.astype(str).str.strip().reset_index(drop=True)


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


def load_three_feats(feat_paths, meta_path, group_col=None):
    meta_df = pd.read_csv(meta_path, dtype={'ID': str, **({group_col: str} if group_col else {})})
    if not {'ID', 'target'}.issubset(meta_df.columns) or meta_df.empty:
        raise ValueError('元数据必须非空并包含 ID 和 target 列')
    if meta_df['ID'].isna().any() or meta_df['ID'].str.strip().eq('').any():
        raise ValueError('元数据 ID 不能为空')
    target = pd.to_numeric(meta_df['target'], errors='raise').astype(np.float32)
    if not np.isfinite(target).all():
        raise ValueError('target 必须为有限数值')
    groups = protein_groups(meta_df, group_col)
    if pd.DataFrame({'ID': meta_df['ID'], 'group': groups}).groupby('ID')['group'].nunique().gt(1).any():
        raise ValueError('同一 ID 不能属于不同蛋白质组')
    if len(feat_paths) != 3:
        raise ValueError('必须提供三个分支特征文件')
    features = [load_single_feat(path, meta_df) for path in feat_paths]
    return features, target, groups, meta_df['ID']


def grouped_splits(groups, n_splits):
    if n_splits < 2 or groups.nunique() < n_splits:
        raise ValueError(f"{n_splits} 折划分需要至少 {n_splits} 个蛋白质组，实际 {groups.nunique()} 个")
    splits = list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))
    for train_idx, held_idx in splits:
        if len(train_idx) < 2:
            raise ValueError('每个训练折至少需要两个样本（BatchNorm 要求）')
        if set(groups.iloc[train_idx]) & set(groups.iloc[held_idx]):
            raise RuntimeError('蛋白质分组泄漏')
    return splits


def scaled_datasets(features, target, train_idx, held_idx):
    train_features, held_features = [], []
    for feature in features:
        scaler = MinMaxScaler()
        train_features.append(pd.DataFrame(scaler.fit_transform(feature.iloc[train_idx])))
        held_features.append(pd.DataFrame(scaler.transform(feature.iloc[held_idx])))
    return (FeatureDataset(*train_features, target.iloc[train_idx]),
            FeatureDataset(*held_features, target.iloc[held_idx]))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrainingBatchSampler:
    """将末尾单样本合并到前一批，避免 BatchNorm 报错且不丢弃样本。"""
    def __init__(self, size, batch_size):
        if size < 2 or batch_size < 2:
            raise ValueError('训练样本数和 batch_size 均须 >= 2')
        self.size, self.batch_size = size, batch_size

    def __iter__(self):
        indices = torch.randperm(self.size).tolist()
        batches = [indices[i:i + self.batch_size] for i in range(0, self.size, self.batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        return iter(batches)

    def __len__(self):
        count = (self.size + self.batch_size - 1) // self.batch_size
        return count - int(count > 1 and self.size % self.batch_size == 1)


def make_model(input_dims, params, device):
    return CombinedMLPMoEModel(
        input_dims=input_dims, d_model=params['d_model'],
        hidden_dim_mlp=params['hidden_dim_mlp'], num_experts=params['num_experts'],
        top_k=params['top_k'], dropout=params['dropout']).to(device)


def evaluate_model(model, data_loader, device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for x1, x2, x3, target in data_loader:
            predictions.append(model(x1.to(device), x2.to(device), x3.to(device)).cpu().numpy().reshape(-1))
            targets.append(target.numpy().reshape(-1))
    predictions, targets = np.concatenate(predictions), np.concatenate(targets)
    if not np.isfinite(predictions).all():
        raise ValueError('模型产生非有限预测值')
    return predictions, targets


class UndefinedCorrelationError(ValueError):
    """No epoch produced a defined validation correlation score."""


def correlation_monitor(truth, prediction, spearman_weight=0.5):
    valid = len(truth) >= 2 and np.ptp(truth) > 0 and np.ptp(prediction) > 0
    rho = float(spearmanr(truth, prediction)[0]) if valid else np.nan
    r = float(pearsonr(truth, prediction)[0]) if valid else np.nan
    score = rho if spearman_weight == 1 else r if spearman_weight == 0 else spearman_weight * rho + (1 - spearman_weight) * r
    return {'spearman': rho, 'pearson': r, 'score': score}


def train_model(model, dataset, params, device, val_dataset=None, fixed_epochs=None, callback=None,
                validation_monitor=None):
    loader = DataLoader(dataset, batch_sampler=TrainingBatchSampler(len(dataset), params['batch_size']))
    val_loader = (DataLoader(val_dataset, batch_size=params['batch_size'])
                  if val_dataset is not None else None)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.L1Loss()
    best_score, best_state, best_epoch, counter = -float('inf'), None, 0, 0
    validation_monitor = validation_monitor or correlation_monitor
    epochs = fixed_epochs if fixed_epochs is not None else params['epochs']
    for epoch in range(1, epochs + 1):
        # 验证会进入 eval 模式；每轮必须恢复 train 模式。
        model.train()
        for x1, x2, x3, target in loader:
            optimizer.zero_grad()
            outputs = model(x1.to(device), x2.to(device), x3.to(device)).reshape(-1)
            loss = criterion(outputs, target.to(device))
            if not torch.isfinite(loss):
                raise ValueError('训练损失非有限，请检查数据和超参数')
            loss.backward()
            optimizer.step()
        if val_loader is not None:
            prediction, truth = evaluate_model(model, val_loader, device)
            validation = validation_monitor(truth, prediction)
            if callback is not None:
                callback(epoch, prediction, truth, validation)
            score = validation['score']
            if np.isfinite(score) and score > best_score:
                best_score, best_epoch, counter = score, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                counter += 1
                if counter >= params['patience']:
                    break
    if val_loader is not None:
        if best_state is None:
            raise UndefinedCorrelationError('所有验证epoch的相关系数均无定义，无法选择最佳权重')
        model.load_state_dict(best_state)
        return best_epoch
    return epochs


def regression_metrics(truth, prediction):
    correlations_valid = len(truth) >= 2 and np.ptp(truth) > 0 and np.ptp(prediction) > 0
    return {
        'r2': metrics.r2_score(truth, prediction) if len(truth) >= 2 and np.ptp(truth) > 0 else np.nan,
        'mae': metrics.mean_absolute_error(truth, prediction),
        'rmse': np.sqrt(metrics.mean_squared_error(truth, prediction)),
        'spearman': spearmanr(truth, prediction)[0] if correlations_valid else np.nan,
        'pearson': pearsonr(truth, prediction)[0] if correlations_valid else np.nan,
        'kendall_t': weightedtau(truth, prediction)[0] if correlations_valid else np.nan,
    }

