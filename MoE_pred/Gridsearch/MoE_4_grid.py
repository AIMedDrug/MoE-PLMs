import argparse
import os
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

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 1. 稀疏MoE模块（保持不变）
class Sparse_MoE(nn.Module):
    def __init__(self, d_model, num_experts=8, top_k=2):
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
        top_k_values, top_k_indices = torch.topk(gate_logits, self.top_k, dim=1)
        gate_weights = torch.softmax(top_k_values, dim=1)
        
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

# 2. MLP特征交互层（修改为四分支）
class MLPInteractionLayer(nn.Module):
    def __init__(self, d_model, hidden_dim=None, dropout=0.4):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim if hidden_dim else 2 * d_model
        
        # 四分支拼接特征（4*d_model）
        self.mlp = nn.Sequential(
            nn.Linear(4 * d_model, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 4 * d_model),
            nn.LayerNorm(4 * d_model)
        )

    def forward(self, x1, x2, x3, x4):
        batch_size = x1.shape[0]
        # 展平特征
        x1_flat = x1.squeeze(1)  # [batch, d_model]
        x2_flat = x2.squeeze(1)
        x3_flat = x3.squeeze(1)
        x4_flat = x4.squeeze(1)
        # 四分支拼接
        combined = torch.cat([x1_flat, x2_flat, x3_flat, x4_flat], dim=1)  # [batch, 4*d_model]
        
        mlp_out = self.mlp(combined)
        
        # 拆分回四分支
        x1_out = mlp_out[:, :self.d_model].unsqueeze(1)
        x2_out = mlp_out[:, self.d_model:2*self.d_model].unsqueeze(1)
        x3_out = mlp_out[:, 2*self.d_model:3*self.d_model].unsqueeze(1)
        x4_out = mlp_out[:, 3*self.d_model:].unsqueeze(1)
        
        return x1_out, x2_out, x3_out, x4_out

# 3. 主模型（修改为四分支）
class CombinedMLPMoEModel(nn.Module):
    def __init__(self, input_dims, d_model=64, hidden_dim_mlp=None, num_experts=8, top_k=2, dropout=0.5):
        super().__init__()
        self.d_model = d_model
        
        # 四分支特征投影
        self.proj_x1 = nn.Linear(input_dims[0], d_model)
        self.proj_x2 = nn.Linear(input_dims[1], d_model)
        self.proj_x3 = nn.Linear(input_dims[2], d_model)
        self.proj_x4 = nn.Linear(input_dims[3], d_model)  # 新增x4投影
        
        # MLP特征交互（四分支）
        self.mlp_interaction = MLPInteractionLayer(d_model, hidden_dim_mlp, dropout)
        
        # 稀疏MoE增强
        self.moe = Sparse_MoE(d_model, num_experts, top_k)
        
        # 特征融合（四分支拼接：4*d_model）
        self.fusion_fc = nn.Linear(d_model * 4, d_model)
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

    def forward(self, x1, x2, x3, x4):
        # 特征投影+增加序列维度
        x1_proj = self.proj_x1(x1).unsqueeze(1)  # [batch, 1, d_model]
        x2_proj = self.proj_x2(x2).unsqueeze(1)
        x3_proj = self.proj_x3(x3).unsqueeze(1)
        x4_proj = self.proj_x4(x4).unsqueeze(1)  # x4投影
        
        # MLP特征交互
        x1_mlp, x2_mlp, x3_mlp, x4_mlp = self.mlp_interaction(x1_proj, x2_proj, x3_proj, x4_proj)
        
        # MoE增强+特征融合
        x1_moe = self.moe(x1_mlp).squeeze(1)  # [batch, d_model]
        x2_moe = self.moe(x2_mlp).squeeze(1)
        x3_moe = self.moe(x3_mlp).squeeze(1)
        x4_moe = self.moe(x4_mlp).squeeze(1)  # x4的MoE处理
        
        # 四分支拼接融合
        fused = torch.cat([x1_moe, x2_moe, x3_moe, x4_moe], dim=1)  # [batch, 4*d_model]
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

# 4. 数据集类（增加x4）
class FeatureDataset(Dataset):
    def __init__(self, x1_feats, x2_feats, x3_feats, x4_feats, target):
        self.x1 = torch.tensor(x1_feats.values, dtype=torch.float32)
        self.x2 = torch.tensor(x2_feats.values, dtype=torch.float32)
        self.x3 = torch.tensor(x3_feats.values, dtype=torch.float32)
        self.x4 = torch.tensor(x4_feats.values, dtype=torch.float32)  # x4特征
        self.target = torch.tensor(target.values, dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.x3[idx], self.x4[idx], self.target[idx]

# 5. 数据加载工具（保留test_meta的使用）
def load_single_feat(feat_path, meta_df):
    """加载单个分支特征，并与元数据ID对齐"""
    if feat_path.endswith('.pkl'):
        embed = pd.read_pickle(feat_path)
        embed_df = pd.DataFrame.from_dict(embed).T.reset_index()
        embed_df.rename(columns={'index': 'ID'}, inplace=True)
    elif feat_path.endswith('.pt'):
        embed = torch.load(feat_path, weights_only=True)
        embed_df = pd.DataFrame.from_dict(embed).T.reset_index()
        embed_df.rename(columns={'index': 'ID'}, inplace=True)
    elif feat_path.endswith('.csv'):
        embed_df = pd.read_csv(feat_path)
    else:
        raise ValueError('仅支持 .pkl, .pt, .csv 格式')
    
    # 与元数据ID对齐（核心：保留元数据的ID匹配逻辑）
    aligned_feat = meta_df[['ID']].merge(embed_df, how='inner', on='ID')
    return aligned_feat.drop(columns=['ID'])

def load_four_feats(feat_paths, meta_path):
    """加载四分支特征，依赖元数据（含ID和target）"""
    meta_df = pd.read_csv(meta_path)
    if 'ID' not in meta_df.columns or 'target' not in meta_df.columns:
        raise ValueError('元数据必须包含 ID 和 target 列（train_meta和test_meta均需）')
    
    # 加载四个分支特征（均通过meta_df对齐ID）
    x1_feat = load_single_feat(feat_paths[0], meta_df)
    x2_feat = load_single_feat(feat_paths[1], meta_df)
    x3_feat = load_single_feat(feat_paths[2], meta_df)
    x4_feat = load_single_feat(feat_paths[3], meta_df)  # 新增x4
    
    # 校验样本数量（与元数据一致）
    assert len(x1_feat) == len(x2_feat) == len(x3_feat) == len(x4_feat) == len(meta_df), \
        "特征文件与元数据样本数量不匹配（请检查ID对齐）"
    return x1_feat, x2_feat, x3_feat, x4_feat, meta_df['target']  # 返回元数据中的target

# 6. 特征归一化（四分支）
def scale_four_features(train_x1, train_x2, train_x3, train_x4, test_x1, test_x2, test_x3, test_x4):
    scaler_x1 = MinMaxScaler(feature_range=(0, 1))
    scaler_x2 = MinMaxScaler(feature_range=(0, 1))
    scaler_x3 = MinMaxScaler(feature_range=(0, 1))
    scaler_x4 = MinMaxScaler(feature_range=(0, 1))  # x4归一化器
    
    train_x1_scaled = scaler_x1.fit_transform(train_x1)
    train_x2_scaled = scaler_x2.fit_transform(train_x2)
    train_x3_scaled = scaler_x3.fit_transform(train_x3)
    train_x4_scaled = scaler_x4.fit_transform(train_x4)
    
    test_x1_scaled = scaler_x1.transform(test_x1)
    test_x2_scaled = scaler_x2.transform(test_x2)
    test_x3_scaled = scaler_x3.transform(test_x3)
    test_x4_scaled = scaler_x4.transform(test_x4)
    
    return (
        pd.DataFrame(train_x1_scaled), pd.DataFrame(train_x2_scaled), 
        pd.DataFrame(train_x3_scaled), pd.DataFrame(train_x4_scaled),
        pd.DataFrame(test_x1_scaled), pd.DataFrame(test_x2_scaled), 
        pd.DataFrame(test_x3_scaled), pd.DataFrame(test_x4_scaled)
    )

# 7. 训练与评估函数（保留test_meta对应的数据加载）
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    best_val_loss = float('inf')
    patience = 20
    counter = 0
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x1_batch, x2_batch, x3_batch, x4_batch, batch_target in train_loader:
            x1_batch, x2_batch, x3_batch, x4_batch = x1_batch.to(device), x2_batch.to(device), x3_batch.to(device), x4_batch.to(device)
            batch_target = batch_target.to(device)
            
            outputs = model(x1_batch, x2_batch, x3_batch, x4_batch)
            loss = criterion(outputs.squeeze(), batch_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            val_loss = evaluate_loss(model, val_loader, criterion, device)
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                counter += 1
                if counter >= patience:
                    print(f"早停于Epoch {epoch+1}（验证集损失连续{patience}轮未下降）")
                    model.load_state_dict(torch.load('best_model.pth'))
                    return

def evaluate_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x1_batch, x2_batch, x3_batch, x4_batch, batch_target in data_loader:
            x1_batch, x2_batch, x3_batch, x4_batch = x1_batch.to(device), x2_batch.to(device), x3_batch.to(device), x4_batch.to(device)
            batch_target = batch_target.to(device)
            outputs = model(x1_batch, x2_batch, x3_batch, x4_batch)

            loss = criterion(outputs.squeeze(), batch_target)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x1_batch, x2_batch, x3_batch, x4_batch, batch_target in data_loader:
            x1_batch, x2_batch, x3_batch, x4_batch = x1_batch.to(device), x2_batch.to(device), x3_batch.to(device), x4_batch.to(device)
            outputs = model(x1_batch, x2_batch, x3_batch, x4_batch)
            
            all_preds.extend(outputs.cpu().numpy().squeeze())
            all_targets.extend(batch_target.numpy())
    return np.array(all_preds), np.array(all_targets)

# 8. 结果保存（不变）
def save_results(r2_train, mae_train, rmse_train, r2_test, mae_test, rmse_test, rho_train, rho_test, pearson_train ,pearson_test,kendall_t_test, method, params):
    result = pd.DataFrame({
        'method': [method],
        'r2_train': [r2_train],
        'mae_train': [mae_train],
        'rmse_train': [rmse_train],
        'spearman_train': [rho_train],
        'pearson_train':[pearson_train],
        'r2_test': [r2_test],
        'mae_test': [mae_test],
        'rmse_test': [rmse_test],
        'spearman_test': [rho_test],
        'pearson_test':[pearson_test],
        'kendall_t':[kendall_t_test]
    })

    # 添加超参数到结果中
    for key, value in params.items():
        result[key] = [value]
    
    return result

# 9. 回归主流程（保留test_meta的target使用）
def run_regression_with_params(train_x1, train_x2, train_x3, train_x4, train_target, test_x1, test_x2, test_x3, test_x4, test_target,params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"当前超参数: {params}")

    # 从训练集拆分验证集（包含x4）
    from sklearn.model_selection import train_test_split
    train_x1, val_x1, train_x2, val_x2, train_x3, val_x3, train_x4, val_x4, train_target, val_target = train_test_split(
        train_x1, train_x2, train_x3, train_x4, train_target, test_size=0.2, random_state=42
    )
    
    # 构建数据集（使用train_target和test_target，均来自对应meta）
    train_dataset = FeatureDataset(train_x1, train_x2, train_x3, train_x4, train_target)
    val_dataset = FeatureDataset(val_x1, val_x2, val_x3, val_x4, val_target)
    test_dataset = FeatureDataset(test_x1, test_x2, test_x3, test_x4, test_target)  # test_target来自test_meta
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
    
    # 初始化四分支模型
    input_dims = [train_x1.shape[1], train_x2.shape[1], train_x3.shape[1], train_x4.shape[1]]
    model = CombinedMLPMoEModel(
        input_dims=input_dims,
        d_model=params['d_model'],
        hidden_dim_mlp=params['hidden_dim_mlp'],
        num_experts=params['num_experts'],
        top_k=params['top_k'],
        dropout=params['dropout']
    ).to(device)
    
    # 损失函数与优化器
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    
    # 训练
    print("\n开始训练...")
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=params['epochs'])
    
    # 评估（使用test_dataset的target，即test_meta中的target）
    y_pred_train, y_true_train = evaluate_model(model, train_loader, device)
    y_pred_val, y_true_val = evaluate_model(model, val_loader, device)
    y_pred_test, y_true_test = evaluate_model(model, test_loader, device)  # 测试集标签来自test_meta
    
    # 计算指标
    metrics_train = {
        'r2': metrics.r2_score(y_true_train, y_pred_train),
        'mae': metrics.mean_absolute_error(y_true_train, y_pred_train),
        'rmse': np.sqrt(metrics.mean_squared_error(y_true_train, y_pred_train)),
        'spearman': spearmanr(y_true_train, y_pred_train)[0],
        'pearson': pearsonr(y_true_train, y_pred_train.squeeze())[0],
        'kendall_t': weightedtau(y_true_train, y_pred_train.squeeze())[0]
    }
    metrics_test = {
        'r2': metrics.r2_score(y_true_test, y_pred_test),  # 依赖test_meta的target
        'mae': metrics.mean_absolute_error(y_true_test, y_pred_test),
        'rmse': np.sqrt(metrics.mean_squared_error(y_true_test, y_pred_test)),
        'spearman': spearmanr(y_true_test, y_pred_test)[0],
        'pearson': pearsonr(y_true_test, y_pred_test.squeeze())[0],
        'kendall_t': weightedtau(y_true_test, y_pred_test)[0]
    }
    
    # 打印结果
    print(f"\n验证集指标 - R²: {metrics.r2_score(y_true_val, y_pred_val):.3f}, MAE: {metrics.mean_absolute_error(y_true_val, y_pred_val):.3f}")
    print(f"训练集指标 - R²: {metrics_train['r2']:.3f}, MAE: {metrics_train['mae']:.3f}, "
          f"RMSE: {metrics_train['rmse']:.3f}, Spearman: {metrics_train['spearman']:.3f}")
    print(f"测试集指标 - R²: {metrics_test['r2']:.3f}, MAE: {metrics_test['mae']:.3f}, "
          f"RMSE: {metrics_test['rmse']:.3f}, Spearman:{metrics_test['spearman']:.3f},pearson:{metrics_test['pearson']:.3f},kendall_t:{metrics_test['kendall_t']:.3f}")
    
    return (metrics_train['r2'], metrics_train['mae'], metrics_train['rmse'],
            metrics_test['r2'], metrics_test['mae'], metrics_test['rmse'],
            metrics_train['spearman'], metrics_test['spearman'],
            metrics_train['pearson'], metrics_test['pearson'],metrics_test['kendall_t'])

# 网格搜索主函数
def grid_search(train_x1, train_x2,train_x3,train_x4, train_target, test_x1, test_x2,test_x3,test_x4, test_target, param_grid):
    # 存储所有结果
    all_results = []
    best_spearman = -float('inf')
    best_params = None
    
    # 生成所有参数组合
    from itertools import product
    
    # 提取参数名称和可能值
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    
    # 遍历所有参数组合
    for i, values in enumerate(product(*param_values)):
        params = dict(zip(param_names, values))
        print(f"\n===== 开始参数组合 {i+1}/{len(list(product(*param_values)))} =====")
        
        # 运行回归
        metrics = run_regression_with_params(
            train_x1, train_x2,train_x3,train_x4, train_target,
            test_x1, test_x2,test_x3,test_x4, test_target,
            params
        )
        
        # 保存结果
        result = save_results(*metrics, method="MLP+MoE(2branches)_grid_search", params=params)
        all_results.append(result)
        
        # 检查是否为最佳结果
        test_spearman = metrics[7]  # 对应metrics_test['spearman']
        if test_spearman > best_spearman:
            best_spearman = test_spearman
            best_params = params
            print(f"找到新的最佳参数组合，Test Spearman: {best_spearman:.4f}")
    
    # 合并所有结果
    all_results_df = pd.concat(all_results, ignore_index=True)
    
    return all_results_df, best_params, best_spearman


# 10. 文件入口流程（明确保留test_meta的使用）
def run_grid_search_on_files(train_feat_paths, train_meta_path, test_feat_paths, test_meta_path, param_grid):
    """
    加载训练/测试数据，其中：
    - train_meta_path：训练集元数据（含ID和target）
    - test_meta_path：测试集元数据（含ID和target）（** 保留test_meta的输入 **）
    """
    method = "MLP+MoE(4branches)"
    
    # 加载训练数据（依赖train_meta）
    print("加载训练数据（使用train_meta对齐ID和target）...")
    train_x1, train_x2, train_x3, train_x4, train_target = load_four_feats(train_feat_paths, train_meta_path)
    
    # 加载测试数据（** 关键：依赖test_meta对齐ID和target **）
    print("加载测试数据（使用test_meta对齐ID和target）...")
    test_x1, test_x2, test_x3, test_x4, test_target = load_four_feats(test_feat_paths, test_meta_path)  # 此处明确使用test_meta
    
    # 特征归一化
    print("特征归一化...")
    scaled = scale_four_features(
        train_x1, train_x2, train_x3, train_x4,
        test_x1, test_x2, test_x3, test_x4
    )
    train_x1_scaled, train_x2_scaled, train_x3_scaled, train_x4_scaled, \
    test_x1_scaled, test_x2_scaled, test_x3_scaled, test_x4_scaled = scaled
    
    # 运行回归（测试集target来自test_meta）
    all_results, best_params, best_spearman = grid_search(
        train_x1_scaled, train_x2_scaled,train_x3_scaled,train_x4_scaled,train_target,
        test_x1_scaled, test_x2_scaled,test_x3_scaled,test_x4_scaled,test_target,
        param_grid
    )
    
    return all_results, best_params, best_spearman

# 11. 命令行入口（** 保留test_meta参数 **）
def main():
    parser = argparse.ArgumentParser(description="四分支特征回归模型（MLP+MoE）")
    # 训练集参数
    parser.add_argument("--train_x1", required=True, help="训练集x1特征文件路径")
    parser.add_argument("--train_x2", required=True, help="训练集x2特征文件路径")
    parser.add_argument("--train_x3", required=True, help="训练集x3特征文件路径")
    parser.add_argument("--train_x4", required=True, help="训练集x4特征文件路径")  # 新增x4
    parser.add_argument("--train_meta", required=True, help="训练集元数据路径（含ID和target）")
    # 测试集参数（** 保留test_meta **）
    parser.add_argument("--test_x1", required=True, help="测试集x1特征文件路径")
    parser.add_argument("--test_x2", required=True, help="测试集x2特征文件路径")
    parser.add_argument("--test_x3", required=True, help="测试集x3特征文件路径")
    parser.add_argument("--test_x4", required=True, help="测试集x4特征文件路径")  # 新增x4
    parser.add_argument("--test_meta", required=True, help="测试集元数据路径（含ID和target）")  # 明确保留test_meta参数
    # 输出参数
    parser.add_argument("-o", "--output", required=True, help="结果输出CSV文件路径")
    
    args = parser.parse_args()
    
    # 定义超参数搜索空间
    param_grid = {
        # 模型结构参数
        'd_model': [32,64,128],
        'hidden_dim_mlp': [32,64,128,256],
        'num_experts': [4,6,8],
        'top_k': [1,2,3],
        'dropout': [0.3,0.5],
        
        # 训练参数
        'batch_size': [16,32],
        'lr': [1e-4],
        'weight_decay': [5e-4],
        'epochs': [100],
        'patience': [20]
    }

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
      # 运行网格搜索并保存结果
    all_results, best_params, best_spearman = run_grid_search_on_files(
        train_feat_paths=[args.train_x1, args.train_x2,args.train_x3,args.train_x4],
        train_meta_path=args.train_meta,
        test_feat_paths=[args.test_x1, args.test_x2,args.test_x3,args.test_x4],
        test_meta_path=args.test_meta,
        param_grid=param_grid
    )
    
    all_results.to_csv(args.output, index=False)
    print(f"\n所有结果已保存至: {args.output}")
    print(f"最佳参数组合: {best_params}")
    print(f"最佳Test Spearman: {best_spearman:.4f}")
    
    # 单独保存最佳参数
    best_params_df = pd.DataFrame([best_params])
    best_params_path = os.path.splitext(args.output)[0] + "moeresult-4-turn-11.11-T5+600M+650M+Amplify350m_best_params_best_params.csv"
    best_params_df.to_csv(best_params_path, index=False)
    print(f"最佳参数已保存至: {best_params_path}")
if __name__ == "__main__":
    main()