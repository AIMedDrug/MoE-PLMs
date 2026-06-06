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
import random
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
    def __init__(self, input_dims, d_model=64, hidden_dim_mlp=None, num_experts=8, top_k=2, dropout=0.5):
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
    # 新增早停机制，防止过拟合
    best_val_loss = float('inf')
    patience = 20  # 连续20轮验证集损失不下降则停止
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
        
        # 每10轮验证一次
        if (epoch + 1) % 10 == 0:
            val_loss = evaluate_loss(model, val_loader, criterion, device)
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
            
            # 早停判断
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), 'best_model.pth')  # 保存最优模型
            else:
                counter += 1
                if counter >= patience:
                    print(f"早停于Epoch {epoch+1}（验证集损失连续{patience}轮未下降）")
                    model.load_state_dict(torch.load('best_model.pth'))  # 加载最优权重
                    return

# 新增验证损失计算函数
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

# 9. 回归主流程（修改模型初始化和训练策略，新增迁移学习加载权重逻辑）
def run_regression(train_x1, train_x2, train_x3, train_target, test_x1, test_x2, test_x3, test_target, params, pretrained_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"当前超参数: {params}")
    # 从训练集拆分验证集（用于早停）
    from sklearn.model_selection import train_test_split
    train_x1, val_x1, train_x2, val_x2, train_x3, val_x3, train_target, val_target = train_test_split(
        train_x1, train_x2, train_x3, train_target, test_size=0.2, random_state=42
    )
    
    # 构建数据集
    train_dataset = FeatureDataset(train_x1, train_x2, train_x3, train_target)
    val_dataset = FeatureDataset(val_x1, val_x2, val_x3, val_target)
    test_dataset = FeatureDataset(test_x1, test_x2, test_x3, test_target)
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
    
    # 初始化MLP+MoE模型（核心修改）
    input_dims = [train_x1.shape[1], train_x2.shape[1], train_x3.shape[1]]
    model = CombinedMLPMoEModel(
         input_dims=input_dims,
        d_model=params['d_model'],
        hidden_dim_mlp=params['hidden_dim_mlp'],
        num_experts=params['num_experts'],
        top_k=params['top_k'],
        dropout=params['dropout']
    ).to(device)
    
    # ===== 迁移学习核心逻辑：加载预训练权重 =====
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            # 加载预训练权重
            pretrained_weights = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(pretrained_weights)
            print(f"✅ 成功加载预训练权重文件: {pretrained_path}")
        except Exception as e:
            print(f"❌ 加载预训练权重失败: {str(e)}，将从头开始训练")
    elif pretrained_path:
        print(f"❌ 预训练权重文件不存在: {pretrained_path}，将从头开始训练")
    
    # 损失函数与优化器（调整学习率和正则化）
    criterion = nn.L1Loss()  # MAE损失更稳健
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])  # 增加权重衰减
    
    # 训练（加入验证集早停）
    print("\n开始训练...")
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=params['epochs'])
    
    # 评估
    y_pred_train, y_true_train = evaluate_model(model, train_loader, device)
    y_pred_val, y_true_val = evaluate_model(model, val_loader, device)
    y_pred_test, y_true_test = evaluate_model(model, test_loader, device)
    
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
        'r2': metrics.r2_score(y_true_test, y_pred_test),
        'mae': metrics.mean_absolute_error(y_true_test, y_pred_test),
        'rmse': np.sqrt(metrics.mean_squared_error(y_true_test, y_pred_test)),
        'spearman': spearmanr(y_true_test, y_pred_test)[0],
        'pearson': pearsonr(y_true_test, y_pred_test.squeeze())[0],
        'kendall_t': weightedtau(y_true_test, y_pred_test.squeeze())[0]
    }
    
    # 打印结果（含验证集）
    print(f"\n验证集指标 - R²: {metrics.r2_score(y_true_val, y_pred_val):.3f}, MAE: {metrics.mean_absolute_error(y_true_val, y_pred_val):.3f}")
    print(f"训练集指标 - R²: {metrics_train['r2']:.3f}, MAE: {metrics_train['mae']:.3f}, "
          f"RMSE: {metrics_train['rmse']:.3f}, Spearman: {metrics_train['spearman']:.3f}")
    print(f"测试集指标 - R²: {metrics_test['r2']:.3f}, MAE: {metrics_test['mae']:.3f}, "
          f"RMSE: {metrics_test['rmse']:.3f}, Spearman: {metrics_test['spearman']:.3f},pearson:{metrics_test['pearson']:.3f},kendall_t:{metrics_test['kendall_t']:.3f}")
    
    return (metrics_train['r2'], metrics_train['mae'], metrics_train['rmse'],
            metrics_test['r2'], metrics_test['mae'], metrics_test['rmse'],
            metrics_train['spearman'], metrics_test['spearman'],
             metrics_train['pearson'], metrics_test['pearson'],metrics_test['kendall_t'])

def grid_search(train_x1, train_x2,train_x3, train_target, test_x1, test_x2, test_x3, test_target, param_grid, pretrained_path=None):
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
        
        # 运行回归（传递预训练权重路径）
        metrics = run_regression(
            train_x1, train_x2, train_x3, train_target,
            test_x1, test_x2, test_x3, test_target,
            params,
            pretrained_path=pretrained_path
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

# 10. 文件入口流程（修改方法名，新增预训练权重参数传递）
def run_grid_search_on_files(train_feat_paths, train_meta_path, test_feat_paths, test_meta_path, param_grid,nums=5,pretrained_path=None):
    method = "MLP+MoE(3branches)"  # 修改方法名标识
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
    
    all_run_results = []  # 存储所有运行的完整结果
    all_run_best_params = []  # 存储每次运行的最佳参数
    all_run_best_spearmans = []  # 存储每次运行的最佳spearman
    final_best_spearman = -float('inf')  # 初始化全局最优spearman
    final_best_params = None  # 初始化全局最优参数
    
    # 多次运行
    for run_idx in range(nums):
        print(f"\n========== 第 {run_idx+1}/{nums} 次运行 ==========")
        #设置随机种子保证可复现
        seed = 42 + run_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # 修正：接收grid_search返回的3个值（所有结果、本次最佳参数、本次最佳spearman）
        run_results, run_best_params, run_best_spearman = grid_search(
            train_x1_scaled, train_x2_scaled, train_x3_scaled, train_target,
            test_x1_scaled, test_x2_scaled, test_x3_scaled, test_target,
            param_grid,
            pretrained_path=pretrained_path
        )
        
        # 收集本次运行的结果
        all_run_results.append(run_results)
        all_run_best_params.append(run_best_params)
        all_run_best_spearmans.append(run_best_spearman)
        
        # 更新全局最优
        if run_best_spearman > final_best_spearman:
            final_best_spearman = run_best_spearman
            final_best_params = run_best_params
            print(f"第 {run_idx+1} 次运行得到全局最优，Test Spearman: {final_best_spearman:.4f}")
        else:
            print(f"第 {run_idx+1} 次运行最佳Test Spearman: {run_best_spearman:.4f}（当前全局最优：{final_best_spearman:.4f}）")
    
    # 合并所有运行的结果（包含所有参数组合的指标）
    all_results_df = pd.concat(all_run_results, ignore_index=True)
    # 为结果添加「运行次数」列，方便区分不同次运行
    all_results_df['run_id'] = np.repeat(range(1, nums+1), [len(df) for df in all_run_results])
    
    # 整理多次运行的最佳参数和最佳spearman
    best_runs_summary = pd.DataFrame({
        'run_id': range(1, nums+1),
        'best_params': all_run_best_params,
        'best_test_spearman': all_run_best_spearmans
    })
    
    print(f"\n========== 所有运行完成 ==========")
    print(f"全局最优Test Spearman: {final_best_spearman:.4f}")
    print(f"全局最优参数: {final_best_params}")
    
    # 返回：所有运行的完整结果、全局最优参数、全局最优spearman、多次运行的最佳结果汇总
    return all_results_df, final_best_params, final_best_spearman, best_runs_summary

# 11. 命令行入口（新增预训练权重参数）
def main():
    parser = argparse.ArgumentParser(description="三分支特征回归模型（MLP+MoE）- 支持迁移学习")
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
    parser.add_argument("--nums", type=int, default=200, help="重复运行次数，取最优结果（默认200）")
    # 迁移学习参数
    parser.add_argument("--pretrained", default=None, help="迁移学习使用的预训练权重文件路径（如best_model.pth）")
    # 输出参数
    parser.add_argument("-o", "--output", required=True, help="结果输出CSV文件路径")
    
    args = parser.parse_args()
    
    # 定义超参数搜索空间
    param_grid = {
        # 模型结构参数
        'd_model': [64],
        'hidden_dim_mlp': [256],
        'num_experts': [32],
        'top_k': [2],
        'dropout': [0.5],
        
        # 训练参数
        'batch_size': [32],
        'lr': [1e-4],
        'weight_decay': [5e-4],
        'epochs': [50],
        'patience': [20]
    }

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 运行并保存结果（传递预训练权重路径）
    all_results, best_params, best_spearman,best_runs_summary  = run_grid_search_on_files(
        train_feat_paths=[args.train_x1, args.train_x2,args.train_x3],
        train_meta_path=args.train_meta,
        test_feat_paths=[args.test_x1, args.test_x2,args.test_x3],
        test_meta_path=args.test_meta,
        param_grid=param_grid,
        nums=args.nums,  # 传递多次运行次数
        pretrained_path=args.pretrained
    )
    
    all_results.to_csv(args.output, index=False)
    print(f"\n所有结果已保存至: {args.output}")
    print(f"最佳参数组合: {best_params}")
    print(f"最佳Test Spearman: {best_spearman:.4f}")
    
    #单独保存最佳参数
    best_params_df = pd.DataFrame([best_params])
    best_params_path = os.path.splitext(args.output)[0] + "tranfer_best_params.csv"
    best_params_df.to_csv(best_params_path, index=False)
    print(f"最佳参数已保存至: {best_params_path}")

    best_runs_summary_path = os.path.splitext(args.output)[0] + "_best_runs_summary.csv"
    best_runs_summary.to_csv(best_runs_summary_path, index=False)
    print(f"多次运行最佳结果汇总已保存至: {best_runs_summary_path}")

if __name__ == "__main__":
    main()

# python scripts/MOE/MoE-transfer3/DTm-transfer.py  --train_x1 DTm/S4346/protT5/result.pt  --train_x2 DTm/S4346/esmc-600m/result.pt  --train_x3 DTm/S4346/esm2-650m/new.pkl   --train_meta DTm/S4346.csv  --test_x1 DTm/S571/protT5/result.pt  --test_x2 DTm/S571/esmc-600m/result.pt   --test_x3 DTm/S571/esm2-650m/new.pkl  --test_meta DTm/S571.csv --pretrained final_best_model.pth -o scripts/MOE/MoE-transfer3/tranfer.csv