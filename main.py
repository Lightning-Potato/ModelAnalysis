# -*- coding: utf-8 -*-
"""
main.py
该文件是整个实验的入口。此版本已升级为使用K折交叉验证，以获得更具统计学意义和可信度的结果。
它负责调用其他模块中的函数来完成数据准备、模型训练、评估和结果分析。
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, \
    roc_auc_score
from sklearn.model_selection import KFold
import os

# 从我们自己创建的模块中导入函数
# 假设这些模块已存在并正确实现
from data_preparation import load_dataset
from models import load_models_and_tokenizers, calculate_perplexity, get_cls_embedding

# 设置随机种子，确保实验结果可复现
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def train_classifier(features, labels, classifier_type='LogisticRegression'):
    """
    使用给定的特征和标签训练分类器模型。
    """
    if classifier_type == 'LogisticRegression':
        classifier = LogisticRegression(random_state=RANDOM_SEED, max_iter=2000)
    elif classifier_type == 'MLPClassifier':
        classifier = MLPClassifier(random_state=RANDOM_SEED, max_iter=2000)
    else:
        raise ValueError("指定的分类器类型无效。")

    classifier.fit(features, labels)
    return classifier


def evaluate_classifier(classifier, features, labels):
    """
    在给定数据集上评估一个已训练的分类器。
    """
    y_pred = classifier.predict(features)
    accuracy = accuracy_score(labels, y_pred)
    precision = precision_score(labels, y_pred, zero_division=0)
    recall = recall_score(labels, y_pred, zero_division=0)
    f1 = f1_score(labels, y_pred, zero_division=0)
    roc_auc = roc_auc_score(labels, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }


def main():
    """
    主函数，执行整个实验流程。
    """
    print("--- AIGC文本检测实验开始（K折交叉验证版） ---")

    # 第1步: 加载所有模型和分词器
    print("正在加载所有模型和分词器...")
    try:
        models_dict = load_models_and_tokenizers()
    except Exception as e:
        print(f"加载模型时出错：{e}")
        print("请确保网络连接正常且模型已正确安装。程序终止。")
        return

    # 第2步: 加载完整数据集
    print("\n--- 正在加载数据集... ---")
    dataframes = {}
    required_files = {
        'full_dataset': 'my_dataset.csv',
        'robustness_translated': 'robustness_translated.csv',
        'robustness_noisy': 'robustness_noisy.csv'
    }

    all_files_exist = True
    for name, path in required_files.items():
        if os.path.exists(path):
            dataframes[name] = load_dataset(path)
            print(f"成功加载 '{path}'，样本量: {len(dataframes[name])}")
        else:
            print(f"警告：数据集文件 '{path}' 未找到。请确保它存在。")
            all_files_exist = False

    if not all_files_exist:
        print("所需的数据集文件缺失。请在运行实验前创建它们。程序终止。")
        return

    full_dataset = dataframes['full_dataset']

    # 第3步: 为完整数据集提取特征
    print("\n--- 正在为完整数据集提取特征... ---")
    full_dataset['perplexity'] = full_dataset['text'].apply(
        lambda x: calculate_perplexity(x, models_dict['gpt2']['model'], models_dict['gpt2']['tokenizer'])
    )
    full_dataset['roberta_features'] = full_dataset['text'].apply(
        lambda x: get_cls_embedding(x, models_dict['roberta']['model'], models_dict['roberta']['tokenizer'])
    )
    full_dataset['bert_features'] = full_dataset['text'].apply(
        lambda x: get_cls_embedding(x, models_dict['bert']['model'], models_dict['bert']['tokenizer'])
    )
    print("特征提取完成。")

    # 第4步: 设置K折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    all_fold_results = []

    print("\n--- 正在执行5折交叉验证... ---")
    fold_count = 1
    for train_index, test_index in kf.split(full_dataset):
        print(f"-> 正在执行第 {fold_count} 折...")

        train_df = full_dataset.iloc[train_index]
        test_df = full_dataset.iloc[test_index]

        labels_train = train_df['label'].values
        labels_test = test_df['label'].values

        # 训练所有分类器
        features_train_perp = train_df['perplexity'].values.reshape(-1, 1)
        classifier_perp = train_classifier(features_train_perp, labels_train, 'LogisticRegression')

        features_train_roberta = np.vstack(train_df['roberta_features'].values)
        classifier_roberta = train_classifier(features_train_roberta, labels_train, 'MLPClassifier')

        features_train_bert = np.vstack(train_df['bert_features'].values)
        classifier_bert = train_classifier(features_train_bert, labels_train, 'MLPClassifier')

        features_train_hybrid = np.hstack([features_train_perp, features_train_roberta])
        classifier_hybrid = train_classifier(features_train_hybrid, labels_train, 'LogisticRegression')

        # 在当前折的测试集上评估
        features_test_perp = test_df['perplexity'].values.reshape(-1, 1)
        results_perp = evaluate_classifier(classifier_perp, features_test_perp, labels_test)
        results_perp.update({'model_name': 'Perplexity', 'dataset': 'Test Set (基准)', 'fold': fold_count})
        all_fold_results.append(results_perp)

        features_test_roberta = np.vstack(test_df['roberta_features'].values)
        results_roberta = evaluate_classifier(classifier_roberta, features_test_roberta, labels_test)
        results_roberta.update({'model_name': 'RoBERTa', 'dataset': 'Test Set (基准)', 'fold': fold_count})
        all_fold_results.append(results_roberta)

        features_test_bert = np.vstack(test_df['bert_features'].values)
        results_bert = evaluate_classifier(classifier_bert, features_test_bert, labels_test)
        results_bert.update({'model_name': 'BERT', 'dataset': 'Test Set (基准)', 'fold': fold_count})
        all_fold_results.append(results_bert)

        features_test_hybrid = np.hstack([features_test_perp, features_test_roberta])
        results_hybrid = evaluate_classifier(classifier_hybrid, features_test_hybrid, labels_test)
        results_hybrid.update(
            {'model_name': 'Hybrid (Perplexity + RoBERTa)', 'dataset': 'Test Set (基准)', 'fold': fold_count})
        all_fold_results.append(results_hybrid)

        fold_count += 1

    # 第5步: 在鲁棒性数据集上评估
    evaluation_datasets = {
        '鲁棒性 (翻译)': dataframes['robustness_translated'],
        '鲁棒性 (噪声)': dataframes['robustness_noisy']
    }

    print("\n--- 正在评估鲁棒性数据集... ---")
    for dataset_name, df_eval in evaluation_datasets.items():
        print(f"-> 正在评估数据集: {dataset_name}...")
        # 鲁棒性数据集的特征提取已在前面完成
        labels_eval = df_eval['label'].values
        features_eval_perp = df_eval['perplexity'].values.reshape(-1, 1)
        features_eval_roberta = np.vstack(df_eval['roberta_features'].values)
        features_eval_bert = np.vstack(df_eval['bert_features'].values)
        features_eval_hybrid = np.hstack([features_eval_perp, features_eval_roberta])

        # 评估模型
        results_perp = evaluate_classifier(classifier_perp, features_eval_perp, labels_eval)
        results_perp.update({'model_name': 'Perplexity', 'dataset': dataset_name, 'fold': 0})  # 鲁棒性数据集不属于任何折，设为0
        all_fold_results.append(results_perp)

        results_roberta = evaluate_classifier(classifier_roberta, features_eval_roberta, labels_eval)
        results_roberta.update({'model_name': 'RoBERTa', 'dataset': dataset_name, 'fold': 0})
        all_fold_results.append(results_roberta)

        results_bert = evaluate_classifier(classifier_bert, features_eval_bert, labels_eval)
        results_bert.update({'model_name': 'BERT', 'dataset': dataset_name, 'fold': 0})
        all_fold_results.append(results_bert)

        results_hybrid = evaluate_classifier(classifier_hybrid, features_eval_hybrid, labels_eval)
        results_hybrid.update({'model_name': 'Hybrid (Perplexity + RoBERTa)', 'dataset': dataset_name, 'fold': 0})
        all_fold_results.append(results_hybrid)

    # 第6步: 汇总并计算均值和标准差
    print("\n--- 正在汇总结果并生成图表... ---")
    results_df = pd.DataFrame(all_fold_results)

    # 仅对基准数据集（Test Set）计算均值和标准差
    final_results = results_df[results_df['dataset'] == 'Test Set (基准)'].groupby('model_name').agg({
        'accuracy': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1_score': ['mean', 'std']
    }).reset_index()

    # 将多级索引转换为单级，便于打印和保存
    final_results.columns = ['_'.join(col).strip() for col in final_results.columns.values]

    # 将鲁棒性数据集的结果附加到最终结果中
    robustness_results = results_df[results_df['dataset'] != 'Test Set (基准)']
    final_results = pd.concat([final_results, robustness_results], ignore_index=True)

    # 打印最终结果表，现在包含均值和标准差
    print("\n--- 最终实验结果总览 (基准数据集为均值±标准差) ---")
    print(final_results)
    final_results.to_csv('final_evaluation_results.csv', index=False, encoding='utf-8')
    print("\n结果已保存到 final_evaluation_results.csv")

    # 创建 F1 值对比图
    plt.figure(figsize=(16, 10))
    sns.barplot(data=final_results, x='model_name', y='f1_score_mean', hue='dataset')
    plt.title('F1 Score Comparison Across Datasets', fontsize=16)
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('F1 值', fontsize=12)
    plt.legend(title='数据集')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('f1_comparison.png')
    print("图表已保存为 f1_comparison.png")
    plt.show()

    # 创建准确率对比图
    plt.figure(figsize=(16, 10))
    sns.barplot(data=final_results, x='model_name', y='accuracy_mean', hue='dataset')
    plt.title('Accuracy Comparison Across Datasets', fontsize=16)
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.legend(title='数据集')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    print("图表已保存为 accuracy_comparison.png")
    plt.show()

    print("\n--- 实验结束 ---")


if __name__ == "__main__":
    main()
