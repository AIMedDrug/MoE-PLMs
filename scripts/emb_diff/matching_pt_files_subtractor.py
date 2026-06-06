import os
from pathlib import Path
import tensor_del  # 替换原pt_del导入为tensor_del

def find_matching_pt_files(folder1, folder2):
    """查找两个文件夹中名字相同的PT文件"""
    # 获取两个文件夹中的所有PT文件
    folder1_pt_files = {f for f in os.listdir(folder1) 
                       if f.endswith('.pt') and os.path.isfile(os.path.join(folder1, f))}
    
    folder2_pt_files = {f for f in os.listdir(folder2) 
                       if f.endswith('.pt') and os.path.isfile(os.path.join(folder2, f))}
    
    # 找到共同的文件名
    common_files = folder1_pt_files & folder2_pt_files
    return sorted(common_files)

def process_matching_files(folder1, folder2, output_folder):
    """处理所有同名PT文件，执行相减操作并保存结果"""
    # 创建输出文件夹（如果不存在）
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 找到所有同名PT文件
    matching_files = find_matching_pt_files(folder1, folder2)
    
    if not matching_files:
        print("没有找到同名的PT文件")
        return
    
    print(f"找到 {len(matching_files)} 个同名PT文件，开始处理...")
    
    # 处理每个同名文件对
    for filename in matching_files:
        file1_path = os.path.join(folder1, filename)
        file2_path = os.path.join(folder2, filename)
        output_path = os.path.join(output_folder, filename)
        
        print(f"处理文件: {filename}")
        
        # 调用tensor_del中的subtract_tensors函数（该函数已包含保存功能）
        result = tensor_del.subtract_tensors(file1_path, file2_path, output_path)
        
        if result is not None:
            print(f"成功保存结果到 {output_path}")
        else:
            print(f"处理文件 {filename} 失败")
    
    print("处理完成")

if __name__ == "__main__":
    # 可以根据需要修改这些路径
    folder1 = "newdata/esmc_600m/rbf1/mut"  # 第一个文件夹路径
    folder2 = "newdata/esmc_600m/rbf1/wt"  # 第二个文件夹路径
    output_folder = "newdata/esmc_600m/rbf1"  # 结果输出文件夹路径
    
    process_matching_files(folder1, folder2, output_folder)

