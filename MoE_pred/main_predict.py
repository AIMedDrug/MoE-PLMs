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
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
        
        
        gate_logits = self.gate(x_flat)
        top_k_values, top_k_indices = torch.topk(gate_logits, self.top_k, dim=1)
        gate_weights = torch.softmax(top_k_values, dim=1)
        
        
        expert_outputs = []
        for i in range(self.num_experts):
            mask = (top_k_indices == i).float()
            expert_out = self.experts[i](x_flat)
            weighted_out = expert_out.unsqueeze(1) * mask.unsqueeze(-1)
            expert_outputs.append(weighted_out)
        
        
        moe_out = torch.sum(torch.stack(expert_outputs), dim=0)
        moe_out = torch.sum(moe_out * gate_weights.unsqueeze(-1), dim=1)
        return moe_out.view(batch_size, seq_len, self.d_model)


class MLPInteractionLayer(nn.Module):
    def __init__(self, d_model, hidden_dim=None, dropout=0.5):  
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim if hidden_dim else 2 * d_model  
        
        
        self.mlp = nn.Sequential(
            # （3*d_model）→ hidden_dim
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
        x1_flat = x1.squeeze(1)  # [batch, d_model]
        x2_flat = x2.squeeze(1)
        x3_flat = x3.squeeze(1)
        combined = torch.cat([x1_flat, x2_flat, x3_flat], dim=1)  # [batch, 3*d_model]
        
        mlp_out = self.mlp(combined) 
        
        
        x1_out = mlp_out[:, :self.d_model].unsqueeze(1)
        x2_out = mlp_out[:, self.d_model:2*self.d_model].unsqueeze(1)
        x3_out = mlp_out[:, 2*self.d_model:].unsqueeze(1)
        
        return x1_out, x2_out, x3_out

# 3. Main Model （MLP Interaction + MoE, Core modification）
class CombinedMLPMoEModel(nn.Module):
    def __init__(self, input_dims, d_model=64, hidden_dim_mlp=None, num_experts=8, top_k=2, dropout=0.5):
        super().__init__()
        self.d_model = d_model
        
        # 3 branches feature projections (keep the same)
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
    
        x1_proj = self.proj_x1(x1).unsqueeze(1)  # [batch, 1, d_model]
        x2_proj = self.proj_x2(x2).unsqueeze(1)
        x3_proj = self.proj_x3(x3).unsqueeze(1)
        
        x1_mlp, x2_mlp, x3_mlp = self.mlp_interaction(x1_proj, x2_proj, x3_proj)
        
        x1_moe = self.moe(x1_mlp).squeeze(1)  # [batch, d_model]
        x2_moe = self.moe(x2_mlp).squeeze(1)
        x3_moe = self.moe(x3_mlp).squeeze(1)
        
        fused = torch.cat([x1_moe, x2_moe, x3_moe], dim=1)  # [batch, 3*d_model]
        fused = self.fusion_fc(fused)  # [batch, d_model]
        fused = self.bn(fused)
        fused = self.dropout(fused)
        
        # Attention [batch, seq_len=1, d_model]
        fused_seq = fused.unsqueeze(1)
        attn_output, _ = self.attention(
            query=fused_seq,
            key=fused_seq,
            value=fused_seq
        )  # [batch, 1, d_model]
        
        # Final output
        out = self.attention_output(attn_output.squeeze(1))  # [batch, 1]
        return out

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
        raise ValueError('Only supported .pkl, .pt, .csv')
    
    aligned_feat = meta_df[['ID']].merge(embed_df, how='inner', on='ID')
    return aligned_feat.drop(columns=['ID'])

def load_three_feats(feat_paths, meta_path):
    meta_df = pd.read_csv(meta_path)
    if 'ID' not in meta_df.columns or 'target' not in meta_df.columns:
        raise ValueError('Metadata must contain ID and target columns')
    
    x1_feat = load_single_feat(feat_paths[0], meta_df)
    x2_feat = load_single_feat(feat_paths[1], meta_df)
    x3_feat = load_single_feat(feat_paths[2], meta_df)
    
    assert len(x1_feat) == len(x2_feat) == len(x3_feat) == len(meta_df), \
        "The number of feature files and metadata samples does not match (please check the ID alignment)"
    return x1_feat, x2_feat, x3_feat, meta_df['target']


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
            
            # 早停判断
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), 'best_model.pth')  
            else:
                counter += 1
                if counter >= patience:
                    print(f"Epoch {epoch+1}（Verification set loss {patience}consecutive rounds without falling）")
                    model.load_state_dict(torch.load('best_model.pth'))  
                    return

# Loss calculation function
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


def run_regression(train_x1, train_x2, train_x3, train_target, test_x1, test_x2, test_x3, test_target, params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    print(f"current hyperparameters: {params}")
   
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
    

    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])  
    
    print("\n start training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=params['epochs'])
    
    
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
    

    print(f"\n Validation set indicator - R²: {metrics.r2_score(y_true_val, y_pred_val):.3f}, MAE: {metrics.mean_absolute_error(y_true_val, y_pred_val):.3f}")
    print(f"Training set indicators - R²: {metrics_train['r2']:.3f}, MAE: {metrics_train['mae']:.3f}, "
          f"RMSE: {metrics_train['rmse']:.3f}, Spearman: {metrics_train['spearman']:.3f}")
    print(f"Test set indicators - R²: {metrics_test['r2']:.3f}, MAE: {metrics_test['mae']:.3f}, "
          f"RMSE: {metrics_test['rmse']:.3f}, Spearman: {metrics_test['spearman']:.3f},pearson:{metrics_test['pearson']:.3f},kendall_t:{metrics_test['kendall_t']:.3f}")
    
    return (metrics_train['r2'], metrics_train['mae'], metrics_train['rmse'],
            metrics_test['r2'], metrics_test['mae'], metrics_test['rmse'],
            metrics_train['spearman'], metrics_test['spearman'],
             metrics_train['pearson'], metrics_test['pearson'],metrics_test['kendall_t'])

def grid_search(train_x1, train_x2,train_x3, train_target, test_x1, test_x2, test_x3, test_target, param_grid):
    
    all_results = []
    best_spearman = -float('inf')
    best_params = None
    
    from itertools import product
    
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    
    for i, values in enumerate(product(*param_values)):
        params = dict(zip(param_names, values))
        print(f"\n===== Parameter combination {i+1}/{len(list(product(*param_values)))} =====")
        
        metrics = run_regression(
            train_x1, train_x2, train_x3, train_target,
            test_x1, test_x2, test_x3, test_target,
            params
        )
        
        # 保存结果
        result = save_results(*metrics, method="MLP+MoE(2branches)_grid_search", params=params)
        all_results.append(result)
        
        test_spearman = metrics[7]  
        if test_spearman > best_spearman:
            best_spearman = test_spearman
            best_params = params
            print(f"Find the new best parameter combination，Test Spearman: {best_spearman:.4f}")
    
    all_results_df = pd.concat(all_results, ignore_index=True)
    
    return all_results_df, best_params, best_spearman

def run_grid_search_on_files(train_feat_paths, train_meta_path, test_feat_paths, test_meta_path, param_grid):
    method = "MLP+MoE(3branches)" 
    
    print("加载训练数据...")
    train_x1, train_x2, train_x3, train_target = load_three_feats(train_feat_paths, train_meta_path)
    print("加载测试数据...")
    test_x1, test_x2, test_x3, test_target = load_three_feats(test_feat_paths, test_meta_path)
    
    print("特征归一化...")
    scaled = scale_three_features(train_x1, train_x2, train_x3, test_x1, test_x2, test_x3)
    train_x1_scaled, train_x2_scaled, train_x3_scaled, test_x1_scaled, test_x2_scaled, test_x3_scaled = scaled
    
    all_results, best_params, best_spearman = grid_search(
        train_x1_scaled, train_x2_scaled,train_x3_scaled, train_target,
        test_x1_scaled, test_x2_scaled, test_x3_scaled ,test_target,
        param_grid
    )
    
    return all_results, best_params, best_spearman

def main():
    parser = argparse.ArgumentParser(description="regression with MLP+MoE（MLP+MoE）")
    
    parser.add_argument("--train_x1", required=True, help="train_x1")
    parser.add_argument("--train_x2", required=True, help="train_x2")
    parser.add_argument("--train_x3", required=True, help="train_x3")
    parser.add_argument("--train_meta", required=True, help="train_meta")
    
    parser.add_argument("--test_x1", required=True, help="test_x1")
    parser.add_argument("--test_x2", required=True, help="test_x2")
    parser.add_argument("--test_x3", required=True, help="test_x3")
    parser.add_argument("--test_meta", required=True, help="test_meta")
    
    parser.add_argument("-o", "--output", required=True, help="output")
    
    args = parser.parse_args()
    
    
    param_grid = {
        'd_model': [32],
        'hidden_dim_mlp': [256],
        'num_experts': [16],
        'top_k': [1],
        'dropout': [0.3],
        
        'batch_size': [16],
        'lr': [1e-4],
        'weight_decay': [5e-4],
        'epochs': [100],
        'patience': [20]
    }
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_results, best_params, best_spearman = run_grid_search_on_files(
        train_feat_paths=[args.train_x1, args.train_x2,args.train_x3],
        train_meta_path=args.train_meta,
        test_feat_paths=[args.test_x1, args.test_x2,args.test_x3],
        test_meta_path=args.test_meta,
        param_grid=param_grid
    )
    
    all_results.to_csv(args.output, index=False)
    print(f"\n results saved to: {args.output}")
    print(f" best params: {best_params}")
    print(f" best spearman: {best_spearman:.4f}")
    

    best_params_df = pd.DataFrame([best_params])
    best_params_path = os.path.splitext(args.output)[0] + ".csv"
    best_params_df.to_csv(best_params_path, index=False)
    print(f"best params saved to: {best_params_path}")

if __name__ == "__main__":
    main()