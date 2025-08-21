# Author: Hu Jia
# Date: 2025.8.16

# -*- coding: utf-8 -*-
"""
main.py
该文件是整个实验的入口，它负责调用其他模块中的函数来完成数据处理、模型评估和结果分析。
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

# 从我们自己创建的模块中导入函数
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
    print(f"--- 正在评估 {model_name} 模型 ---")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=RANDOM_SEED
    )

    # 选择并训练分类器
    if classifier_type == 'LogisticRegression':
        classifier = LogisticRegression(random_state=RANDOM_SEED, max_iter=2000)
    elif classifier_type == 'MLPClassifier':
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

    print(f"{model_name} 准确率: {accuracy:.4f}")
    print(f"{model_name} 精确率: {precision:.4f}")
    print(f"{model_name} 召回率: {recall:.4f}")
    print(f"{model_name} F1值: {f1:.4f}")

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
    models_dict = load_models_and_tokenizers()

    # 第2步: 准备数据集
    try:
        data = load_dataset('my_dataset.csv')
    except FileNotFoundError:
        print("警告：数据集文件 'my_dataset.csv' 未找到。正在创建模拟数据集。")
        data = create_mock_dataset(RANDOM_SEED, num_samples_per_class=50)

    if data.empty:
        print("数据集为空，无法进行实验。请检查您的数据文件。")
        return

    # 第3步: 特征提取
    print("\n--- 正在提取所有模型的特征 ---")

    data['perplexity'] = data['text'].apply(
        lambda x: calculate_perplexity(x, models_dict['gpt2']['model'], models_dict['gpt2']['tokenizer'])
    )
    data['roberta_features'] = data['text'].apply(
        lambda x: get_cls_embedding(x, models_dict['roberta']['model'], models_dict['roberta']['tokenizer'])
    )
    data['bert_features'] = data['text'].apply(
        lambda x: get_cls_embedding(x, models_dict['bert']['model'], models_dict['bert']['tokenizer'])
    )

    # 由于您注释掉了DeBERTa，我们也移除其特征提取
    # data['deberta_features'] = data['text'].apply(
    #     lambda x: get_cls_embedding(x, models_dict['deberta']['model'], models_dict['deberta']['tokenizer'])
    # )

    print("所有特征提取完成。")

    # 第4步: 评估所有模型
    results = []
    labels = data['label'].values

    # 评估统计模型 (Perplexity)
    results.append(evaluate_model(
        data['perplexity'].values.reshape(-1, 1), labels, 'Perplexity'
    ))

    # 评估深度学习模型
    results.append(evaluate_model(
        np.vstack(data['roberta_features'].values), labels, 'RoBERTa'
    ))
    results.append(evaluate_model(
        np.vstack(data['bert_features'].values), labels, 'BERT'
    ))

    # 由于您注释掉了DeBERTa，我们也移除其评估
    # results.append(evaluate_model(
    #     np.vstack(data['deberta_features'].values), labels, 'DeBERTa'
    # ))

    # 评估混合模型
    hybrid_features_roberta = np.hstack(
        [data['perplexity'].values.reshape(-1, 1), np.vstack(data['roberta_features'].values)])
    results.append(evaluate_model(
        hybrid_features_roberta, labels, 'Hybrid (Perplexity + RoBERTa)'
    ))

    # 由于您注释掉了DeBERTa，我们也移除其混合模型评估
    # hybrid_features_deberta = np.hstack([data['perplexity'].values.reshape(-1, 1), np.vstack(data['deberta_features'].values)])
    # results.append(evaluate_model(
    #     hybrid_features_deberta, labels, 'Hybrid (Perplexity + DeBERTa)'
    # ))

    # 第5步: 结果可视化
    print("\n--- 正在生成结果图表... ---")
    results_df = pd.DataFrame(results)

    # 创建性能指标对比图
    metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    plt.figure(figsize=(14, 8))
    for metric in metrics:
        plt.plot(results_df['model_name'], results_df[metric], marker='o', label=metric)
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    print("图表已保存为 performance_comparison.png")
    plt.show()

    print("\n--- 论文实验结束 ---")


if __name__ == "__main__":
    main()
