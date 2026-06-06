import os
import pandas as pd

def fasta_file_maker_from_csv(csv_file_path, filename):
    """
    从CSV文件读取数据并转换为FASTA格式文件，序列每60个字符自动换行
    
    函数会从CSV文件中读取'ID'和'mut_seqs'列，然后生成FASTA文件，格式为：
    - 第一行为以'>'开头的ID
    - 后续行为对应的突变序列，每60个字符自动换行
    
    参数:
      - csv_file_path (str): CSV文件的路径，该文件必须包含'ID'和'mut_seqs'列
      - filename (str): 要创建的fasta文件名称（不含".fasta"后缀）
    
    返回:
      - 无返回值，直接生成文件
    
    异常:
      - FileNotFoundError: 如果指定的CSV文件不存在
      - ValueError: 如果CSV文件缺少必要的列或包含无效数据
      - OSError: 如果文件写入过程中出现系统错误
    """
    # 检查CSV文件是否存在
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"指定的CSV文件不存在: {csv_file_path}")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        raise ValueError(f"读取CSV文件时发生错误: {str(e)}")
    
    # 检查必要的列是否存在
    required_columns = ['ID', 'mut_seqs']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"CSV文件缺少必要的列: {', '.join(missing)}")
    
    # 检查ID和序列是否为字符串类型
    if not pd.api.types.is_string_dtype(df['ID']):
        raise ValueError("'ID'列必须包含字符串类型数据")
    if not pd.api.types.is_string_dtype(df['mut_seqs']):
        raise ValueError("'wt_seqs'列必须包含字符串类型数据")
    
    # 确保输出目录存在
    output_dir = "DTm"
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建完整文件路径
    file_path = os.path.join(output_dir, f"{filename}_mut.fasta")
    
    # 处理序列换行的辅助函数
    def wrap_sequence(sequence, width=60):
        """将序列按指定宽度换行"""
        return '\n'.join([sequence[i:i+width] for i in range(0, len(sequence), width)])
    
    # 写入FASTA文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                identifier = row['ID']
                # 移除序列中可能存在的空白字符
                clean_sequence = ''.join(str(row['mut_seqs']).split())
                # 按60个字符换行处理
                wrapped_sequence = wrap_sequence(clean_sequence)
                f.write(f'>{identifier}\n{wrapped_sequence}\n')
        print(f"FASTA文件已成功创建: {file_path}")
    except OSError as e:
        raise OSError(f"写入文件时发生错误: {str(e)}")
    

fasta_file_maker_from_csv("DTm/S571.csv", "s571")