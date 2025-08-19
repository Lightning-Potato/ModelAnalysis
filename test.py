# 统计模型：gpt2-large

# 深度学习模型：bert-base-uncased、roberta-base、deberta-v3-base

# 混合模型构建方法：
# 使用 gpt2-large 计算测试文本的 Perplexity 分数。
# 使用 roberta-base 或 deberta-v3-base 提取测试文本的特征向量（例如，取[CLS] token的嵌入向量）。
# 将这两个特征拼接起来，形成一个新的特征向量。
# 训练一个简单的分类器（如sklearn中的LogisticRegression或MLPClassifier）来完成最终的分类任务。


# -*- coding: utf-8 -*-
"""
这是一个为您的论文实验定制的完整Python脚本。
它整合了您提供的模型列表和方法，涵盖了数据准备、模型评估和混合模型构建。
您可以将此代码作为您的研究起点，并用您实际收集的数据替换模拟数据。
"""

# 导入所有必要的库
## 深度学习框架
import torch

## 预训练模型库
# 我们导入了用于不同任务的模型和分词器。
# AutoTokenizer 和 AutoModel 可以根据模型名称自动加载相应的类。
from transformers import (
    AutoTokenizer,
    AutoModel,
    GPT2LMHeadModel
)

## 数据处理与分析
import pandas as pd
import numpy as np

## 机器学习和绘图工具
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子，确保实验结果可复现
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- 第1步: 模型加载和工具初始化 ---
# 加载您大纲中提到的所有模型和分词器
# 确保在运行前已安装所有必需的库

# GPT-2 for Perplexity and AIGC text generation
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
# 需要使用 GPT2LMHeadModel 来计算 perplexity，因为它包含语言模型头
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2-large')
gpt2_model.eval()

# RoBERTa for feature extraction
roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
roberta_model = AutoModel.from_pretrained('roberta-base')
roberta_model.eval()

# BERT for feature extraction
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')
bert_model.eval()

# DeBERTa for feature extraction
deberta_tokenizer = AutoTokenizer.from_pretrained('deberta-v3-base')
deberta_model = AutoModel.from_pretrained('deberta-v3-base')
deberta_model.eval()

# 设置设备为GPU（如果可用），否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_model.to(device)
roberta_model.to(device)
bert_model.to(device)
deberta_model.to(device)

# --- 第2步: 数据集准备 (模拟) ---
# 由于没有实际数据集，我们创建一个模拟数据集来演示流程。
# 在您的实际论文中，您需要用您的真实数据集来替换这部分。

# 模拟真实文本（human_texts）
human_texts = [
    "The advancements in artificial intelligence are reshaping various industries, from healthcare to finance.",
    "A recent study published in Nature reveals new insights into the behavior of black holes.",
    "The old man's weary eyes scanned the horizon, a silent testament to a lifetime of struggle.",
    "For the past three years, our company has seen a steady increase in sales and market share.",
    "Quantum computing promises to solve problems that are currently intractable for classical computers."
]

# 使用 gpt2_model 模拟生成 AIGC 文本
# 注意：在您的实际工作中，您将使用 GPT-3.5 或其他大模型 API
from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='gpt2-large', device=0 if torch.cuda.is_available() else -1)
set_seed(RANDOM_SEED)

aigc_texts = [
    t['generated_text'] for t in
    generator("Hello, I'm a language model,", max_length=100, num_return_sequences=len(human_texts))
]

# 将模拟数据构建成 Pandas DataFrame
data = pd.DataFrame({
    'text': human_texts + aigc_texts,
    'label': [0] * len(human_texts) + [1] * len(aigc_texts)  # 0 for human, 1 for AIGC
})
print("模拟数据集已创建，共计 %d 条样本。" % len(data))


# --- 第3步: 模型实现函数 ---

def calculate_perplexity(text, model, tokenizer, device):
    """
    计算给定文本的困惑度（Perplexity）。
    """
    try:
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
        input_ids = encoded_input['input_ids']

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
            return perplexity.item()
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return None


def get_cls_embedding(text, model, tokenizer, device):
    """
    提取文本的 [CLS] token 嵌入，用作句子特征向量。
    """
    try:
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**encoded_input)
            # [CLS] token的嵌入通常在 outputs.last_hidden_state 的第一个位置
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            return cls_embedding.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


# --- 第4步: 实验流程 ---

if __name__ == "__main__":
    print("\n--- 开始实验 ---")

    # 1. 统计模型 (Perplexity) 评估
    print("\n1. 统计模型评估 (Perplexity)")
    data['perplexity'] = data['text'].apply(
        lambda x: calculate_perplexity(x, gpt2_model, gpt2_tokenizer, device)
    )
    print("困惑度计算完成。")
    print(data[['text', 'label', 'perplexity']].head())

    # 可视化困惑度分布
    plt.figure(figsize=(10, 6))
    sns.histplot(data, x='perplexity', hue='label', bins=20, kde=True, palette=['#1f77b4', '#ff7f0e'])
    plt.title('Distribution of Perplexity Scores')
    plt.xlabel('Perplexity')
    plt.ylabel('Frequency')
    plt.show()

    # 2. 深度学习模型 (RoBERTa) 特征提取
    print("\n2. 深度学习模型特征提取 (RoBERTa)")
    data['roberta_features'] = data['text'].apply(
        lambda x: get_cls_embedding(x, roberta_model, roberta_tokenizer, device)
    )
    print("RoBERTa特征提取完成。")

    # 3. 混合模型构建与评估
    print("\n3. 混合模型构建与评估")
    # 准备特征和标签
    perplexity_feature = data['perplexity'].values.reshape(-1, 1)
    roberta_features_matrix = np.vstack(data['roberta_features'].values)

    # 拼接特征
    hybrid_features = np.hstack([perplexity_feature, roberta_features_matrix])
    labels = data['label'].values

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        hybrid_features, labels, test_size=0.2, random_state=RANDOM_SEED
    )

    # 训练逻辑回归分类器
    classifier = LogisticRegression(random_state=RANDOM_SEED)
    classifier.fit(X_train, y_train)

    # 在测试集上进行预测和评估
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"混合模型（Perplexity + RoBERTa）准确率: {accuracy:.4f}")
    print("\n分类报告:\n", report)

    print("\n--- 实验结束 ---")

