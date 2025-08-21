# -*- coding: utf-8 -*-
"""
robustness_test.py
该脚本用于执行论文大纲中定义的鲁棒性测试。
它将从 my_dataset.csv 中加载数据，划分测试集，并对测试集进行
改写、翻译、噪声干扰，然后重新评估模型的性能。

此版本使用 Hugging Face 的本地开源模型，完全避免了 API 配额和费用问题。
"""
import pandas as pd
import numpy as np
import os
import random
import re
import sys
import torch
import warnings
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 忽略来自 transformers 的一些警告
warnings.filterwarnings("ignore", category=UserWarning)

# 导入 main.py 中的相关函数
try:
    from main import evaluate_model
    from models import load_models_and_tokenizers, calculate_perplexity, get_cls_embedding
except ImportError as e:
    print(f"❌ 导入错误: 无法找到 main.py 或 models.py 中的模块。请确保这些文件在当前目录下。详细错误: {e}")
    sys.exit(1)

# ========== 配置 ==========
DATASET_FILE = "my_dataset.csv"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 检测可用的设备，优先使用 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ 当前设备: {device.upper()}")


# ========== 模型加载函数 ==========

def load_local_models_for_robustness():
    """
    加载用于鲁棒性测试的本地模型。
    """
    print("\n--- 正在加载用于鲁棒性测试的本地模型... ---")

    # 用于文本改写的模型
    rewrite_model_name = "google/flan-t5-base"
    rewrite_tokenizer = AutoTokenizer.from_pretrained(rewrite_model_name)
    rewrite_model = AutoModelForSeq2SeqLM.from_pretrained(rewrite_model_name).to(device)
    print(f"✅ 加载 {rewrite_model_name} 模型成功。")

    # 用于中译英的模型
    zh_en_model_name = "Helsinki-NLP/opus-mt-zh-en"
    zh_en_tokenizer = AutoTokenizer.from_pretrained(zh_en_model_name)
    zh_en_model = AutoModelForSeq2SeqLM.from_pretrained(zh_en_model_name).to(device)
    print(f"✅ 加载 {zh_en_model_name} 模型成功。")

    # 用于英译中的模型
    en_zh_model_name = "Helsinki-NLP/opus-mt-en-zh"
    en_zh_tokenizer = AutoTokenizer.from_pretrained(en_zh_model_name)
    en_zh_model = AutoModelForSeq2SeqLM.from_pretrained(en_zh_model_name).to(device)
    print(f"✅ 加载 {en_zh_model_name} 模型成功。")

    return {
        "rewrite_tokenizer": rewrite_tokenizer,
        "rewrite_model": rewrite_model,
        "zh_en_tokenizer": zh_en_tokenizer,
        "zh_en_model": zh_en_model,
        "en_zh_tokenizer": en_zh_tokenizer,
        "en_zh_model": en_zh_model,
    }


# ========== 鲁棒性测试函数 ==========

def rewrite_text(text, models):
    """
    使用本地模型对文本进行改写，保留语义但改变句式。
    """
    print(f"-> 正在改写文本：'{text[:30]}...'")
    prompt = f"请改写以下文本，使其保持原意但句式和表达方式不同：\n\n{text}"
    inputs = models["rewrite_tokenizer"](prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

    # 使用 .generate 方法进行文本生成
    outputs = models["rewrite_model"].generate(
        **inputs,
        max_length=512,
        num_beams=4,
        early_stopping=True,
        temperature=0.8
    )

    rewritten_text = models["rewrite_tokenizer"].decode(outputs[0], skip_special_tokens=True)
    print("✅ 文本改写成功。")
    return rewritten_text


def translate_and_back(text, models):
    """
    将文本翻译成英文再回译成中文，以改变表达。
    """
    print(f"-> 正在进行翻译回译：'{text[:30]}...'")

    # 中文 -> 英文
    zh_en_inputs = models["zh_en_tokenizer"](text, return_tensors="pt", max_length=512, truncation=True).to(device)
    zh_en_outputs = models["zh_en_model"].generate(**zh_en_inputs)
    translated_en = models["zh_en_tokenizer"].decode(zh_en_outputs[0], skip_special_tokens=True)

    # 英文 -> 中文
    en_zh_inputs = models["en_zh_tokenizer"](translated_en, return_tensors="pt", max_length=512, truncation=True).to(
        device)
    en_zh_outputs = models["en_zh_model"].generate(**en_zh_inputs)
    back_translated_zh = models["en_zh_tokenizer"].decode(en_zh_outputs[0], skip_special_tokens=True)

    print("✅ 翻译回译成功。")
    return back_translated_zh


def add_noise(text, noise_ratio=0.1):
    """
    在文本中随机插入无意义字符。
    """
    print(f"-> 正在对文本添加噪声：'{text[:30]}...'")
    noise_chars = "#@!%&*"
    text_list = list(text)
    num_to_add = int(len(text_list) * noise_ratio)
    for _ in range(num_to_add):
        if not text_list:
            break
        insert_pos = random.randint(0, len(text_list))
        text_list.insert(insert_pos, random.choice(noise_chars))
    print("✅ 噪声添加完成。")
    return "".join(text_list)


# ========== 主函数 ==========
def main():
    """
    执行鲁棒性测试，并生成结果。
    """
    print("--- 鲁棒性测试开始 ---")
    try:
        df = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        print(f"❌ 错误: 文件 {DATASET_FILE} 未找到。请确保该文件存在于当前目录中。")
        return

    # 划分训练集和测试集
    _, df_test, _, _ = train_test_split(df['text'], df['label'], test_size=0.2, random_state=RANDOM_SEED)
    df_test = pd.DataFrame({'text': df_test, 'label': df['label'].loc[df_test.index]})

    # 加载用于评估的原始模型
    base_models_dict = load_models_and_tokenizers()
    # 加载用于鲁棒性测试的本地模型
    local_models = load_local_models_for_robustness()

    num_samples = len(df_test)
    if num_samples < 30:
        print("⚠️ 警告：测试集太小，鲁棒性测试结果可能不准确。建议增加测试样本数量。")
        return

    num_per_group = num_samples // 3
    df_rewrite = df_test.iloc[:num_per_group].copy()
    df_translate = df_test.iloc[num_per_group:2 * num_per_group].copy()
    df_noise = df_test.iloc[2 * num_per_group:].copy()

    all_results = []

    # 1. 文本改写测试
    print("\n--- 正在进行文本改写测试... ---")
    df_rewrite['processed_text'] = df_rewrite['text'].apply(lambda x: rewrite_text(x, local_models))
    df_rewrite['perplexity'] = df_rewrite['processed_text'].apply(
        lambda x: calculate_perplexity(x, base_models_dict['gpt2']['model'], base_models_dict['gpt2']['tokenizer'])
    )
    df_rewrite['roberta_features'] = df_rewrite['processed_text'].apply(
        lambda x: get_cls_embedding(x, base_models_dict['roberta']['model'], base_models_dict['roberta']['tokenizer'])
    )
    all_results.append(evaluate_model(df_rewrite['perplexity'].values.reshape(-1, 1), df_rewrite['label'].values,
                                      'Perplexity (Rewritten)'))
    all_results.append(evaluate_model(np.vstack(df_rewrite['roberta_features'].values), df_rewrite['label'].values,
                                      'RoBERTa (Rewritten)'))

    # 2. 翻译回译测试
    print("\n--- 正在进行翻译回译测试... ---")
    df_translate['processed_text'] = df_translate['text'].apply(lambda x: translate_and_back(x, local_models))
    df_translate['perplexity'] = df_translate['processed_text'].apply(
        lambda x: calculate_perplexity(x, base_models_dict['gpt2']['model'], base_models_dict['gpt2']['tokenizer'])
    )
    df_translate['roberta_features'] = df_translate['processed_text'].apply(
        lambda x: get_cls_embedding(x, base_models_dict['roberta']['model'], base_models_dict['roberta']['tokenizer'])
    )
    all_results.append(evaluate_model(df_translate['perplexity'].values.reshape(-1, 1), df_translate['label'].values,
                                      'Perplexity (Translated)'))
    all_results.append(evaluate_model(np.vstack(df_translate['roberta_features'].values), df_translate['label'].values,
                                      'RoBERTa (Translated)'))

    # 3. 噪声干扰测试
    print("\n--- 正在进行噪声干扰测试... ---")
    df_noise['processed_text'] = df_noise['text'].apply(add_noise)
    df_noise['perplexity'] = df_noise['processed_text'].apply(
        lambda x: calculate_perplexity(x, base_models_dict['gpt2']['model'], base_models_dict['gpt2']['tokenizer'])
    )
    df_noise['roberta_features'] = df_noise['processed_text'].apply(
        lambda x: get_cls_embedding(x, base_models_dict['roberta']['model'], base_models_dict['roberta']['tokenizer'])
    )
    all_results.append(
        evaluate_model(df_noise['perplexity'].values.reshape(-1, 1), df_noise['label'].values, 'Perplexity (Noisy)'))
    all_results.append(
        evaluate_model(np.vstack(df_noise['roberta_features'].values), df_noise['label'].values, 'RoBERTa (Noisy)'))

    # 打印和保存所有结果
    results_df = pd.DataFrame(all_results)
    print("\n--- 所有鲁棒性测试结果 ---")
    print(results_df[['model_name', 'accuracy', 'f1_score']])
    results_df.to_csv('robustness_results.csv', index=False)
    print("✅ 鲁棒性测试结果已保存到 robustness_results.csv。")

    print("\n--- 鲁棒性测试完成 ---")


if __name__ == "__main__":
    main()
