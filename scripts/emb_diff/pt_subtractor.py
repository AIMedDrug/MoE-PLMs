import torch
import os
from pathlib import Path

def load_pt_file(file_path):
    """加载PT文件并返回内容"""
    try:
        data = torch.load(file_path)
        return data
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {str(e)}")
        return None

def save_pt_file(data, file_path):
    """保存PT文件，保持与原文件相同的格式"""
    try:
        # 保持与原文件相同的保存格式
        torch.save(data, file_path)
        return True
    except Exception as e:
        print(f"保存文件 {file_path} 时出错: {str(e)}")
        return False

def subtract_pt_files(file1_path, file2_path):
    """
    对两个PT文件进行相减操作
    返回 file1_data - file2_data 的结果
    """
    data1 = load_pt_file(file1_path)
    data2 = load_pt_file(file2_path)
    
    if data1 is None or data2 is None:
        return None
    
    try:
        # 检查数据类型是否兼容
        if type(data1) != type(data2):
            print(f"文件 {file1_path} 和 {file2_path} 的数据类型不匹配")
            return None
            
        # 如果是字典类型，递归处理每个键
        if isinstance(data1, dict):
            result = {}
            for key in data1:
                if key not in data2:
                    print(f"警告: 键 {key} 在 {file2_path} 中不存在，将被跳过")
                    continue
                result[key] = data1[key] - data2[key]
            return result
        # 如果是张量，直接相减
        elif isinstance(data1, torch.Tensor):
            return data1 - data2
        # 处理其他可能的类型
        else:
            try:
                return data1 - data2
            except Exception as e:
                print(f"相减操作失败: {str(e)}")
                return None
                
    except Exception as e:
        print(f"处理文件 {file1_path} 和 {file2_path} 时出错: {str(e)}")
        return None
