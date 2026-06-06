import pickle
import argparse
import os
import numpy as np
from typing import Dict, Tuple
import torch

def load_pkl_data(file_path: str) -> Dict[str, np.ndarray]:
    """加载pkl文件并验证数据格式为ID到1D numpy数组的字典"""
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 加载pkl文件
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            #data = torch.load(f, weights_only=True)
    except pickle.UnpicklingError:
        raise ValueError(f"pkl文件损坏，无法解析: {file_path}")
    except Exception as e:
        raise RuntimeError(f"加载文件时发生错误: {str(e)}")
    

    # 验证是否为字典
    if not isinstance(data, dict):
        raise TypeError(f"pkl文件内容必须是字典，当前类型: {type(data).__name__}")
    
    valid_data = {}
    invalid_ids = []
    
    # 验证每个ID对应的值是否为1D数值数组
    for id_str, value in data.items():
        # 尝试转换为numpy数组
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (ValueError, TypeError):
            invalid_ids.append(f"{id_str}（无法转换为数值数组）")
            continue
        
        # 验证是否为1D数组
        if arr.ndim != 1:
            invalid_ids.append(f"{id_str}（{arr.ndim}D数组，需要1D数组）")
            continue
        
        # 验证数组非空
        if len(arr) == 0:
            invalid_ids.append(f"{id_str}（空数组）")
            continue
        
        valid_data[id_str] = arr
    
    # 输出无效ID警告
    if invalid_ids:
        print(f"警告: {file_path} 中跳过 {len(invalid_ids)} 个无效ID（示例: {', '.join(invalid_ids[:2])}...）")
    
    # 检查是否有有效数据
    if not valid_data:
        raise ValueError(f"{file_path} 中没有有效的ID-1D数组对")
    
    print(f"成功加载 {file_path}: {len(valid_data)} 个有效条目（数组长度: {len(next(iter(valid_data.values())))}）")
    return valid_data

def subtract_arrays(data1: Dict[str, np.ndarray], data2: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], dict]:
    """对两个字典中相同ID的1D数组执行元素级相减（data1 - data2）"""
    # 获取ID集合
    ids1 = set(data1.keys())
    ids2 = set(data2.keys())
    common_ids = ids1 & ids2
    only_in1 = ids1 - ids2
    only_in2 = ids2 - ids1
    
    # 初始化结果和统计信息
    result = {}
    stats = {
        "common_ids": len(common_ids),
        "success": 0,
        "shape_mismatch": 0,
        "only_in_file1": len(only_in1),
        "only_in_file2": len(only_in2)
    }
    
    # 处理共同ID
    if common_ids:
        print(f"\n处理 {len(common_ids)} 个共同ID的数组相减...")
        for id_str in common_ids:
            arr1 = data1[id_str]
            arr2 = data2[id_str]
            
            # 检查数组形状是否匹配
            if arr1.shape != arr2.shape:
                stats["shape_mismatch"] += 1
                print(f"警告: ID {id_str} 形状不匹配（{arr1.shape} vs {arr2.shape}），已跳过")
                continue
            
            # 执行相减
            result[id_str] = arr1 - arr2
            stats["success"] += 1
    
    # 输出统计结果
    print("\n处理统计:")
    print(f"  共同ID总数: {stats['common_ids']}")
    print(f"  成功相减: {stats['success']}")
    print(f"  形状不匹配跳过: {stats['shape_mismatch']}")
    print(f"  仅在file1中的ID: {stats['only_in_file1']}")
    print(f"  仅在file2中的ID: {stats['only_in_file2']}")
    
    return result, stats

def save_results(result: Dict[str, np.ndarray], output_path: str) -> None:
    """将相减结果保存为pkl文件"""
    try:
        # 创建输出目录（如果需要）
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"\n结果已保存到: {output_path}（包含 {len(result)} 个条目）")
    except Exception as e:
        raise RuntimeError(f"保存结果失败: {str(e)}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="对两个pkl文件中相同ID的1D numpy数组执行元素级相减（file1 - file2）")
    parser.add_argument("file1", help="被减数pkl文件路径")
    parser.add_argument("file2", help="减数pkl文件路径")
    parser.add_argument("output", help="输出结果pkl文件路径")
    args = parser.parse_args()
    
    try:
        # 主流程
        data1 = load_pkl_data(args.file1)
        data2 = load_pkl_data(args.file2)
        result, _ = subtract_arrays(data1, data2)
        save_results(result, args.output)
        print("\n处理完成！")
    except Exception as e:
        print(f"\n错误: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
