# Author: Hu Jia
# Date: 2025.8.16

# -*- coding: utf-8 -*-
"""
data_preparation.py
该文件包含了用于创建或加载数据集的函数。
"""
import pandas as pd
import os
from transformers import pipeline, set_seed
import torch


def load_dataset(file_path):
    """
    加载您的真实数据集。
    假设数据集是一个CSV文件，包含 'text' 和 'label' 两列。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件未找到：{file_path}")

    print(f"--- 正在从 {file_path} 加载数据集... ---")
    data = pd.read_csv(file_path)
    print(f"数据集已加载，共计 {len(data)} 条样本。")
    return data


def create_mock_dataset(random_seed, num_samples_per_class=5):
    """
    如果您的真实数据集不可用，此函数会创建一个模拟数据集用于测试。
    """
    print("--- 正在创建模拟数据集... ---")

    # 模拟真实文本（human_texts）
    # 为了避免ValueError，我们生成与aigc_texts相同数量的模拟文本
    human_texts = [
        f"This is a human-written text sample {i + 1} for demonstration. " +
        "It is a short paragraph to test the model's capabilities."
        for i in range(num_samples_per_class)
    ]

    # 使用 gpt2-large 模拟生成 AIGC 文本
    generator = pipeline('text-generation', model='gpt2-large', device=0 if torch.cuda.is_available() else -1)
    set_seed(random_seed)

    aigc_texts = [
        t['generated_text'] for t in generator(
            "Hello, I'm a language model, and I'm generating a short paragraph for you.",
            max_length=100, num_return_sequences=num_samples_per_class
        )
    ]

    # 将模拟数据构建成 Pandas DataFrame
    data = pd.DataFrame({
        'text': human_texts + aigc_texts,
        'label': [0] * num_samples_per_class + [1] * num_samples_per_class  # 0 for human, 1 for AIGC
    })
    print(f"模拟数据集已创建，共计 {len(data)} 条样本。")
    return data
