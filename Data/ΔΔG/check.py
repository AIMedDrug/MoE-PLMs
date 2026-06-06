import os
import pickle
import torch

def check_feature_lengths(feat_path):
    """
    检查特征文件中所有样本的特征长度是否一致
    
    参数:
        feat_path: 特征文件路径（支持.pkl或.pt格式）
    """
    # 检查文件是否存在
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"文件不存在: {feat_path}")
    
    # 加载特征数据
    try:
        if feat_path.endswith('.pkl'):
            with open(feat_path, 'rb') as f:
                feat_dict = pickle.load(f)  # 格式: {样本ID: 特征数组, ...}
        elif feat_path.endswith('.pt'):
            feat_dict = torch.load(feat_path, weights_only=True)  # 安全加载PyTorch文件
        else:
            raise ValueError(f"不支持的文件格式: {feat_path}，仅支持.pkl和.pt")
    except Exception as e:
        raise RuntimeError(f"加载文件失败: {str(e)}")
    
    # 检查是否为字典格式
    if not isinstance(feat_dict, dict):
        raise TypeError("特征文件内容必须是字典格式（键为样本ID，值为特征数组）")
    
    # 提取所有样本的特征长度
    length_info = []
    for sample_id, feat in feat_dict.items():
        # 处理numpy数组或torch张量
        if isinstance(feat, torch.Tensor):
            feat_len = len(feat.flatten())  # 展平张量后取长度
        else:
            feat_len = len(feat)  # 默认为numpy数组或列表
        length_info.append((sample_id, feat_len))
    
    # 检查长度是否一致
    all_lengths = [l for _, l in length_info]
    unique_lengths = set(all_lengths)
    
    if len(unique_lengths) == 1:
        print(f"✅ 所有样本特征长度一致，均为: {unique_lengths.pop()}")
    else:
        print(f"❌ 发现{len(unique_lengths)}种不同的特征长度: {sorted(unique_lengths)}")
        print("\n详细样本及长度：")
        for sample_id, length in length_info:
            print(f"样本ID: {sample_id} → 特征长度: {length}")

if __name__ == "__main__":
    # 替换为你的特征文件路径（支持.pkl或.pt）
    FEATURE_FILE_PATH = "newdata/esm2_3B/new.pkl"  # 例如: "train_x2.pkl"
    
    # 执行检查
    check_feature_lengths(FEATURE_FILE_PATH)