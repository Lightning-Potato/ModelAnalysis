# Author: Hu Jia
# Date: 2025.8.16
# Updated: 2025.8.22

# -*- coding: utf-8 -*-
"""
main.py
该文件是整个实验的入口，它负责调用其他模块中的函数来完成数据处理、模型评估和结果分析。
此版本在原有基础上增加了跨领域性能分析和鲁棒性测试功能。
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, \
    roc_auc_score
from datetime import datetime
import os

# 从我们自己创建的模块中导入函数
# 假设这些模块已存在并正确工作
from data_preparation import load_dataset, create_mock_dataset
from models import load_models_and_tokenizers, calculate_perplexity, get_cls_embedding

# 设置随机种子，确保实验结果可复现
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def evaluate_model(features, labels, model_name, classifier_type='LogisticRegression'):
    """
    使用给定的特征和标签评估分类器模型。
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=RANDOM_SEED
    )

    # 选择并训练分类器
    if classifier_type == 'LogisticRegression':
        classifier = LogisticRegression(random_state=RANDOM_SEED, max_iter=2000)
    elif classifier_type == 'MLPClassifier':
        # 修正: 'RANDM_SEED' -> 'RANDOM_SEED'
        classifier = MLPClassifier(random_state=RANDOM_SEED, max_iter=2000)
    else:
        raise ValueError("Invalid classifier type specified.")

    classifier.fit(X_train, y_train)

    # 在测试集上进行预测和评估
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'report': classification_report(y_test, y_pred, output_dict=True)
    }


def main():
    """
    主函数，执行整个实验流程。
    """
    print("--- 论文实验开始 ---")

    # 第1步: 加载所有模型和工具
    print("正在加载所有模型和分词器...")
    try:
        models_dict = load_models_and_tokenizers()
    except Exception as e:
        print(f"加载模型时出错：{e}")
        print("请确保网络连接正常且模型已正确安装。程序终止。")
        return

    # 第2步: 准备数据集
    # 按照大纲要求，分领域加载数据集
    dataset_paths = {
        # '新闻文本': 'news_data.csv',
        # '学术文本': 'academic_data.csv',
        # '小说文本': 'novel_data.csv',
        # '鲁棒性-改写': 'rewritten_text.csv',
        '鲁棒性-翻译': 'translated_text.csv',
        '鲁棒性-噪声': 'noisy_text.csv',
    }

    datasets = {}
    for name, path in dataset_paths.items():
        if os.path.exists(path):
            datasets[name] = load_dataset(path)
            print(f"成功加载 {name} 数据集，样本量: {len(datasets[name])}")
        else:
            print(f"警告：数据集文件 '{path}' 未找到。跳过该部分实验。")

    if not datasets:
        print("未找到任何数据集，正在创建模拟数据集用于演示。")
        datasets['模拟数据集'] = create_mock_dataset(RANDOM_SEED, num_samples_per_class=50)

    # 第3步: 评估所有模型
    all_results = []

    # 循环遍历所有数据集进行评估
    for dataset_name, data in datasets.items():
        print(f"\n--- 正在评估数据集: {dataset_name} ---")

        if data.empty:
            print(f"数据集 {dataset_name} 为空，无法进行评估。")
            continue

        # 特征提取
        print("正在提取所有模型的特征...")
        data['perplexity'] = data['text'].apply(
            lambda x: calculate_perplexity(x, models_dict['gpt2']['model'], models_dict['gpt2']['tokenizer'])
        )
        data['roberta_features'] = data['text'].apply(
            lambda x: get_cls_embedding(x, models_dict['roberta']['model'], models_dict['roberta']['tokenizer'])
        )
        data['bert_features'] = data['text'].apply(
            lambda x: get_cls_embedding(x, models_dict['bert']['model'], models_dict['bert']['tokenizer'])
        )
        print("所有特征提取完成。")

        # 评估模型
        labels = data['label'].values

        # 统计模型 (Perplexity)
        results = evaluate_model(data['perplexity'].values.reshape(-1, 1), labels, 'Perplexity')
        results['dataset'] = dataset_name
        all_results.append(results)

        # 深度学习模型
        results = evaluate_model(np.vstack(data['roberta_features'].values), labels, 'RoBERTa')
        results['dataset'] = dataset_name
        all_results.append(results)

        results = evaluate_model(np.vstack(data['bert_features'].values), labels, 'BERT')
        results['dataset'] = dataset_name
        all_results.append(results)

        # 混合模型
        hybrid_features = np.hstack(
            [data['perplexity'].values.reshape(-1, 1), np.vstack(data['roberta_features'].values)])
        results = evaluate_model(hybrid_features, labels, 'Hybrid (Perplexity + RoBERTa)')
        results['dataset'] = dataset_name
        all_results.append(results)

    # 第4步: 结果汇总与可视化
    print("\n--- 正在汇总结果并生成图表... ---")
    results_df = pd.DataFrame(all_results)
    results_df = results_df.round(4)  # 保留4位小数

    # 打印最终结果表
    print("\n--- 实验结果总览 ---")
    print(results_df[['dataset', 'model_name', 'accuracy', 'precision', 'recall', 'f1_score']])
    results_df.to_csv('evaluation_results.csv', index=False, encoding='utf-8')
    print("\n实验结果已保存到 evaluation_results.csv")

    # 创建性能指标对比图
    # 按数据集分组绘制F1值
    plt.figure(figsize=(16, 10))
    sns.barplot(data=results_df, x='model_name', y='f1_score', hue='dataset')
    plt.title('F1 Score Comparison Across Datasets', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.legend(title='Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('f1_comparison.png')
    print("图表已保存为 f1_comparison.png")
    plt.show()

    # 创建一个简单的准确率对比图
    plt.figure(figsize=(16, 10))
    sns.barplot(data=results_df, x='model_name', y='accuracy', hue='dataset')
    plt.title('Accuracy Comparison Across Datasets', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(title='Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    print("图表已保存为 accuracy_comparison.png")
    plt.show()

    print("\n--- 论文实验结束 ---")


if __name__ == "__main__":
    main()

