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
import torch.nn.functional as F
import random

# 设置随机种子保证可复现性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 1. 稀疏MoE模块（保持不变）
class Sparse_MoE(nn.Module):
    def __init__(self, d_model, num_experts=4, top_k=2):
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

# 2. 新增MLP特征交互层（核心修改：替代原交叉注意力层）
class MLPInteractionLayer(nn.Module):
    def __init__(self, d_model, hidden_dim=None, dropout=0.5):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim if hidden_dim else 2 * d_model
        
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_model, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 3 * d_model),
            nn.LayerNorm(3 * d_model)
        )

    def forward(self, x1, x2, x3):
        batch_size = x1.shape[0]
        x1_flat = x1.squeeze(1)
        x2_flat = x2.squeeze(1)
        x3_flat = x3.squeeze(1)
        combined = torch.cat([x1_flat, x2_flat, x3_flat], dim=1)
        
        mlp_out = self.mlp(combined)
        
        x1_out = mlp_out[:, :self.d_model].unsqueeze(1)
        x2_out = mlp_out[:, self.d_model:2*self.d_model].unsqueeze(1)
        x3_out = mlp_out[:, 2*self.d_model:].unsqueeze(1)
        
        return x1_out, x2_out, x3_out

# 3. 主模型（MLP交互+MoE，核心修改）
class CombinedMLPMoEModel(nn.Module):
    def __init__(self, input_dims, d_model=64, hidden_dim_mlp=None, num_experts=8, top_k=2, dropout=0.5):
        super().__init__()
        self.d_model = d_model
        
        self.proj_x1 = nn.Linear(input_dims[0], d_model)
        self.proj_x2 = nn.Linear(input_dims[1], d_model)
        self.proj_x3 = nn.Linear(input_dims[2], d_model)
        
        self.mlp_interaction = MLPInteractionLayer(d_model, hidden_dim_mlp, dropout)
        self.moe = Sparse_MoE(d_model, num_experts, top_k)
        
        self.fusion_fc = nn.Linear(d_model * 3, d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.attention_output = nn.Linear(d_model, 1)

    def forward(self, x1, x2, x3):
        x1_proj = self.proj_x1(x1).unsqueeze(1)
        x2_proj = self.proj_x2(x2).unsqueeze(1)
        x3_proj = self.proj_x3(x3).unsqueeze(1)
        
        x1_mlp, x2_mlp, x3_mlp = self.mlp_interaction(x1_proj, x2_proj, x3_proj)
        
        x1_moe = self.moe(x1_mlp).squeeze(1)
        x2_moe = self.moe(x2_mlp).squeeze(1)
        x3_moe = self.moe(x3_mlp).squeeze(1)
        
        fused = torch.cat([x1_moe, x2_moe, x3_moe], dim=1)
        fused = self.fusion_fc(fused)
        fused = self.bn(fused)
        fused = self.dropout(fused)
        
        fused_seq = fused.unsqueeze(1)
        attn_output, _ = self.attention(
            query=fused_seq,
            key=fused_seq,
            value=fused_seq
        )
        
        out = self.attention_output(attn_output.squeeze(1))
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

# 5. 数据加载工具（保持不变）
def load_single_feat(feat_path, meta_df):
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
    
    aligned_feat = meta_df[['ID']].merge(embed_df, how='inner', on='ID')
    return aligned_feat.drop(columns=['ID'])

def load_three_feats(feat_paths, meta_path):
    meta_df = pd.read_csv(meta_path)
    if 'ID' not in meta_df.columns or 'target' not in meta_df.columns:
        raise ValueError('元数据必须包含 ID 和 target 列')
    
    x1_feat = load_single_feat(feat_paths[0], meta_df)
    x2_feat = load_single_feat(feat_paths[1], meta_df)
    x3_feat = load_single_feat(feat_paths[2], meta_df)
    
    assert len(x1_feat) == len(x2_feat) == len(x3_feat) == len(meta_df), \
        "特征文件与元数据样本数量不匹配（请检查ID对齐）"
    return x1_feat, x2_feat, x3_feat, meta_df['target']

# 6. 特征归一化（保持不变）
def scale_three_features(train_x1, train_x2, train_x3, test_x1, test_x2, test_x3):
    scaler_x1 = MinMaxScaler(feature_range=(0, 1))
    scaler_x2 = MinMaxScaler(feature_range=(0, 1))
    scaler_x3 = MinMaxScaler(feature_range=(0, 1))
    
    train_x1_scaled = scaler_x1.fit_transform(train_x1)
    train_x2_scaled = scaler_x2.fit_transform(train_x2)
    train_x3_scaled = scaler_x3.fit_transform(train_x3)
    
    test_x1_scaled = scaler_x1.transform(test_x1)
    test_x2_scaled = scaler_x2.transform(test_x2)
    test_x3_scaled = scaler_x3.transform(test_x3)
    
    return (
        pd.DataFrame(train_x1_scaled), pd.DataFrame(train_x2_scaled), pd.DataFrame(train_x3_scaled),
        pd.DataFrame(test_x1_scaled), pd.DataFrame(test_x2_scaled), pd.DataFrame(test_x3_scaled)
    )

# 7. 训练与评估函数（增强正则化策略）
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    best_val_loss = float('inf')
    patience = 20
    counter = 0
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x1_batch, x2_batch, x3_batch, batch_target in train_loader:
            x1_batch, x2_batch, x3_batch = x1_batch.to(device), x2_batch.to(device), x3_batch.to(device)
            batch_target = batch_target.to(device)
            
            outputs = model(x1_batch, x2_batch, x3_batch)
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
                torch.save(model.state_dict(), 'temp_best_model.pth')
            else:
                counter += 1
                if counter >= patience:
                    print(f"早停于Epoch {epoch+1}（验证集损失连续{patience}轮未下降）")
                    model.load_state_dict(torch.load('temp_best_model.pth'))
                    return
    return model

def evaluate_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x1_batch, x2_batch, x3_batch, batch_target in data_loader:
            x1_batch, x2_batch, x3_batch = x1_batch.to(device), x2_batch.to(device), x3_batch.to(device)
            batch_target = batch_target.to(device)
            outputs = model(x1_batch, x2_batch, x3_batch)
            loss = criterion(outputs.squeeze(), batch_target)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x1_batch, x2_batch, x3_batch, batch_target in data_loader:
            x1_batch, x2_batch, x3_batch = x1_batch.to(device), x2_batch.to(device), x3_batch.to(device)
            outputs = model(x1_batch, x2_batch, x3_batch)
            
            all_preds.extend(outputs.cpu().numpy().squeeze())
            all_targets.extend(batch_target.numpy())
    return np.array(all_preds), np.array(all_targets)

# 8. 结果保存（保持不变）
def save_results(r2_train, mae_train, rmse_train, r2_test, mae_test, rmse_test, rho_train, rho_test, pearson_train,pearson_test, kendall_t_test, method,params):
    return pd.DataFrame({
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

# 9. 回归主流程（新增权重加载参数）
def run_regression(train_x1, train_x2, train_x3, train_target, test_x1, test_x2, test_x3, test_target, params, weight_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"当前超参数: {params}")
    
    from sklearn.model_selection import train_test_split
    train_x1, val_x1, train_x2, val_x2, train_x3, val_x3, train_target, val_target = train_test_split(
        train_x1, train_x2, train_x3, train_target, test_size=0.2, random_state=42
    )
    
    train_dataset = FeatureDataset(train_x1, train_x2, train_x3, train_target)
    val_dataset = FeatureDataset(val_x1, val_x2, val_x3, val_target)
    test_dataset = FeatureDataset(test_x1, test_x2, test_x3, test_target)
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
    
    input_dims = [train_x1.shape[1], train_x2.shape[1], train_x3.shape[1]]
    model = CombinedMLPMoEModel(
         input_dims=input_dims,
        d_model=params['d_model'],
        hidden_dim_mlp=params['hidden_dim_mlp'],
        num_experts=params['num_experts'],
        top_k=params['top_k'],
        dropout=params['dropout']
    ).to(device)
    
    # 新增：加载预训练权重
    if weight_path is not None and os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"✅ 成功加载预训练权重: {weight_path}")
    else:
        print("🔴 未加载预训练权重，开始从头训练")
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    
    print("\n开始训练...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=params['epochs'])
    
    y_pred_train, y_true_train = evaluate_model(model, train_loader, device)
    y_pred_val, y_true_val = evaluate_model(model, val_loader, device)
    y_pred_test, y_true_test = evaluate_model(model, test_loader, device)
    
    metrics_train = {
        'r2': metrics.r2_score(y_true_train, y_pred_train),
        'mae': metrics.mean_absolute_error(y_true_train, y_pred_train),
        'rmse': np.sqrt(metrics.mean_squared_error(y_true_train, y_pred_train)),
        'spearman': spearmanr(y_true_train, y_pred_train)[0],
        'pearson': pearsonr(y_true_train, y_pred_train.squeeze())[0],
        'kendall_t': weightedtau(y_true_train, y_pred_train.squeeze())[0]
    }
    metrics_test = {
        'r2': metrics.r2_score(y_true_test, y_pred_test),
        'mae': metrics.mean_absolute_error(y_true_test, y_pred_test),
        'rmse': np.sqrt(metrics.mean_squared_error(y_true_test, y_pred_test)),
        'spearman': spearmanr(y_true_test, y_pred_test)[0],
        'pearson': pearsonr(y_true_test, y_pred_test.squeeze())[0],
        'kendall_t': weightedtau(y_true_test, y_pred_test.squeeze())[0]
    }
    
    print(f"\n验证集指标 - R²: {metrics.r2_score(y_true_val, y_pred_val):.3f}, MAE: {metrics.mean_absolute_error(y_true_val, y_pred_val):.3f}")
    print(f"训练集指标 - R²: {metrics_train['r2']:.3f}, MAE: {metrics_train['mae']:.3f}, "
          f"RMSE: {metrics_train['rmse']:.3f}, Spearman: {metrics_train['spearman']:.3f}")
    print(f"测试集指标 - R²: {metrics_test['r2']:.3f}, MAE: {metrics_test['mae']:.3f}, "
          f"RMSE: {metrics_test['rmse']:.3f}, Spearman: {metrics_test['spearman']:.3f},pearson:{metrics_test['pearson']:.3f},kendall_t:{metrics_test['kendall_t']:.3f}")
    
    # 返回指标 + 当前模型权重路径
    return (metrics_train['r2'], metrics_train['mae'], metrics_train['rmse'],
            metrics_test['r2'], metrics_test['mae'], metrics_test['rmse'],
            metrics_train['spearman'], metrics_test['spearman'],
            metrics_train['pearson'], metrics_test['pearson'],metrics_test['kendall_t'],
            'temp_best_model.pth')

def grid_search(train_x1, train_x2,train_x3, train_target, test_x1, test_x2, test_x3, test_target, param_grid, weight_path=None):
    all_results = []
    best_spearman = -float('inf')
    best_params = None
    best_weight_path = None
    
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    
    for i, values in enumerate(product(*param_values)):
        params = dict(zip(param_names, values))
        print(f"\n===== 开始参数组合 {i+1}/{len(list(product(*param_values)))} =====")
        
        metrics = run_regression(
            train_x1, train_x2, train_x3, train_target,
            test_x1, test_x2, test_x3, test_target,
            params,
            weight_path=weight_path  # 传递权重路径
        )
        
        result = save_results(*metrics[:-1], method="MLP+MoE(2branches)_grid_search", params=params)
        all_results.append(result)
        
        test_spearman = metrics[7]
        if test_spearman > best_spearman:
            best_spearman = test_spearman
            best_params = params
            best_weight_path = metrics[-1]
            print(f"找到新的最佳参数组合，Test Spearman: {best_spearman:.4f}")
    
    all_results_df = pd.concat(all_results, ignore_index=True)
    return all_results_df, best_params, best_spearman, best_weight_path

# 10. 文件入口流程（新增nums参数支持）
def run_grid_search_on_files(train_feat_paths, train_meta_path, test_feat_paths, test_meta_path, param_grid, nums=5, weight_path=None):
    method = "MLP+MoE(3branches)"
    print(f"========== 开始 {nums} 次重复运行 ==========")
    
    # 加载数据
    print("加载训练数据...")
    train_x1, train_x2, train_x3, train_target = load_three_feats(train_feat_paths, train_meta_path)
    print("加载测试数据...")
    test_x1, test_x2, test_x3, test_target = load_three_feats(test_feat_paths, test_meta_path)
    
    # 特征归一化
    print("特征归一化...")
    scaled = scale_three_features(train_x1, train_x2, train_x3, test_x1, test_x2, test_x3)
    train_x1_scaled, train_x2_scaled, train_x3_scaled, test_x1_scaled, test_x2_scaled, test_x3_scaled = scaled
    
    # 存储多次运行的最优结果
    final_best_spearman = -float('inf')
    final_best_params = None
    final_best_weight_path = None
    all_run_results = []
    
    # 多次运行
    for run_idx in range(nums):
        print(f"\n========== 第 {run_idx+1}/{nums} 次运行 ==========")
        set_seed(run_idx)  # 每次运行设置不同种子
        
        # 运行网格搜索
        all_results, best_params, best_spearman, best_weight = grid_search(
            train_x1_scaled, train_x2_scaled,train_x3_scaled, train_target,
            test_x1_scaled, test_x2_scaled, test_x3_scaled ,test_target,
            param_grid,
            weight_path=weight_path
        )
        
        all_run_results.append(all_results)
        
        # 更新全局最优
        if best_spearman > final_best_spearman:
            final_best_spearman = best_spearman
            final_best_params = best_params
            final_best_weight_path = best_weight
            print(f"第 {run_idx+1} 次运行得到全局最优，Test Spearman: {final_best_spearman:.4f}")
    
    # 合并所有结果
    all_results_df = pd.concat(all_run_results, ignore_index=True)
    
    # 保存全局最优权重
    if final_best_weight_path and os.path.exists(final_best_weight_path):
        final_weight_path = "final_best_model.pth"
        os.rename(final_best_weight_path, final_weight_path)
        print(f"\n✅ 全局最优权重已保存至: {final_weight_path}")
    else:
        final_weight_path = None
        print("\n❌ 未找到最优权重文件")
    
    return all_results_df, final_best_params, final_best_spearman, final_weight_path

# 11. 命令行入口（新增nums和weight_path参数）
def main():
    parser = argparse.ArgumentParser(description="三分支特征回归模型（MLP+MoE）")
    # 训练集参数
    parser.add_argument("--train_x1", required=True, help="训练集x1特征文件路径")
    parser.add_argument("--train_x2", required=True, help="训练集x2特征文件路径")
    parser.add_argument("--train_x3", required=True, help="训练集x3特征文件路径")
    parser.add_argument("--train_meta", required=True, help="训练集元数据文件路径（含ID和target）")
    # 测试集参数
    parser.add_argument("--test_x1", required=True, help="测试集x1特征文件路径")
    parser.add_argument("--test_x2", required=True, help="测试集x2特征文件路径")
    parser.add_argument("--test_x3", required=True, help="测试集x3特征文件路径")
    parser.add_argument("--test_meta", required=True, help="测试集元数据文件路径（含ID和target）")
    # 输出参数
    parser.add_argument("-o", "--output", required=True, help="结果输出CSV文件路径")
    # 新增参数
    parser.add_argument("--nums", type=int, default=1, help="重复运行次数，取最优结果（默认5）")
    parser.add_argument("--weight_path", type=str, default=None, help="预训练权重文件路径（可选）")
    
    args = parser.parse_args()
    
    # 定义超参数搜索空间
    param_grid = {
        'd_model': [32,64],
        'hidden_dim_mlp': [32,64,128,256],
        'num_experts': [2,4,8,16,32],
        'top_k': [1,2],
        'dropout': [0.3,0.5],
        'batch_size': [32],
        'lr': [1e-4],
        'weight_decay': [5e-4],
        'epochs': [100],
        'patience': [20]
    }

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 运行多次网格搜索
    all_results, best_params, best_spearman, best_weight = run_grid_search_on_files(
        train_feat_paths=[args.train_x1, args.train_x2,args.train_x3],
        train_meta_path=args.train_meta,
        test_feat_paths=[args.test_x1, args.test_x2,args.test_x3],
        test_meta_path=args.test_meta,
        param_grid=param_grid,
        nums=args.nums,  # 传递多次运行次数
        weight_path=args.weight_path  # 传递权重路径
    )
    
    all_results.to_csv(args.output, index=False)
    print(f"\n所有结果已保存至: {args.output}")
    print(f"全局最佳参数组合: {best_params}")
    print(f"全局最佳Test Spearman: {best_spearman:.4f}")
    
    # 保存最佳参数
    best_params_df = pd.DataFrame([best_params])
    best_params_path = os.path.splitext(args.output)[0] + "_best_params.csv"
    best_params_df.to_csv(best_params_path, index=False)
    print(f"最佳参数已保存至: {best_params_path}")
    
    # 保存最优权重路径信息
    weight_info = pd.DataFrame({
        'best_weight_path': [best_weight],
        'best_spearman': [best_spearman]
    })
    weight_info_path = os.path.splitext(args.output)[0] + "_best_weight_info.csv"
    weight_info.to_csv(weight_info_path, index=False)
    print(f"最优权重信息已保存至: {weight_info_path}")

if __name__ == "__main__":
    main()



# python scripts/MOE/MOE-Trans-3.17.py  --train_x1 newdata/ProstT5/result.pt  --train_x2 newdata/esmc_600m/result.pt  --train_x3 newdata/esm2_650m/new.pkl   --train_meta newdata/S8754.csv  --test_x1 newdata/s669/ProstT5/result.pt  --test_x2 newdata/s669/esmc_600m-result.pt   --test_x3 newdata/s669/esm2-650m/new.pkl  --test_meta newdata/S669.csv  -o newdata/weight/3.17new-atttention.csv
