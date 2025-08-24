# Author: Hu Jia
# Date: 2025.8.24

import pandas as pd
from sklearn.model_selection import train_test_split
import os


# 注释：这个脚本用于将纯净的原始数据集划分为训练集和测试集。
# 训练集用于模型微调，测试集用于评估模型的基础性能。
# 划分比例为 80% 训练集，20% 测试集。

def split_dataset(input_file, train_output_file, test_output_file, test_size=0.2, random_state=42):
    """
    将数据集划分为训练集和测试集并保存为新的CSV文件。

    Args:
        input_file (str): 原始数据集文件路径，例如 'my_dataset.csv'。
        train_output_file (str): 训练集保存路径，例如 'train.csv'。
        test_output_file (str): 测试集保存路径，例如 'test.csv'。
        test_size (float): 测试集所占比例，默认为 0.2 (20%)。
        random_state (int): 随机种子，确保每次划分结果一致，便于复现。
    """
    try:
        # Step 1: 读取原始数据集
        print(f"正在读取数据集: {input_file}")
        df = pd.read_csv(input_file)

        # 检查DataFrame是否为空
        if df.empty:
            print("错误：数据集为空，请检查文件内容。")
            return

        # 检查DataFrame是否有'text'和'label'列
        if 'text' not in df.columns or 'label' not in df.columns:
            print("错误：数据集中缺少必要的 'text' 或 'label' 列。")
            print("当前列为:", df.columns.tolist())
            return

        # Step 2: 随机划分训练集和测试集
        # 使用 stratify 参数可以确保训练集和测试集中，'label' 的分布比例与原始数据集保持一致。
        # 这一步非常重要，可以避免因为随机划分导致类别不平衡。
        print(f"正在按 {1 - test_size:.0%} / {test_size:.0%} 的比例划分数据集...")
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['label']
        )

        # Step 3: 将划分后的数据集保存为新的CSV文件
        train_df.to_csv(train_output_file, index=False)
        test_df.to_csv(test_output_file, index=False)

        print("-" * 30)
        print("数据集划分完成！")
        print(f"训练集已保存至: {train_output_file} (样本数: {len(train_df)})")
        print(f"测试集已保存至: {test_output_file} (样本数: {len(test_df)})")
        print("-" * 30)

    except FileNotFoundError:
        print(f"错误：文件 {input_file} 未找到。请确保文件路径正确。")
    except Exception as e:
        print(f"在划分数据集时发生错误: {e}")


if __name__ == '__main__':
    # 假设你的原始数据集文件名为 'my_dataset.csv'，且在同一目录下
    input_csv = 'my_dataset.csv'

    # 划分后将生成的训练集和测试集文件名
    train_csv = 'train.csv'
    test_csv = 'test.csv'

    split_dataset(input_csv, train_csv, test_csv)

