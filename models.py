# -*- coding: utf-8 -*-
"""
models.py
该文件包含了所有与模型相关的函数，用于加载预训练模型、计算困惑度和提取特征。
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    GPT2LMHeadModel
)

# 设置设备为GPU（如果可用），否则使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models_and_tokenizers():
    """
    加载所有需要的预训练模型和分词器。
    """
    print("--- 正在加载模型和分词器... ---")

    # GPT-2 for Perplexity and AIGC text generation
    print("正在加载 gpt2-large 模型...")
    gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2-large').to(DEVICE)
    gpt2_model.eval()

    # RoBERTa for feature extraction
    print("正在加载 roberta-base 模型...")
    roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    roberta_model = AutoModel.from_pretrained('roberta-base').to(DEVICE)
    roberta_model.eval()

    # BERT for feature extraction
    print("正在加载 bert-base-uncased 模型...")
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased').to(DEVICE)
    bert_model.eval()

    # # DeBERTa for feature extraction
    # print("正在加载 deberta-v3-base 模型...")
    # deberta_tokenizer = AutoTokenizer.from_pretrained('deberta-v3-base')
    # deberta_model = AutoModel.from_pretrained('deberta-v3-base').to(DEVICE)
    # deberta_model.eval()

    print("所有模型加载完成。")

    return {
        'gpt2': {'model': gpt2_model, 'tokenizer': gpt2_tokenizer},
        'roberta': {'model': roberta_model, 'tokenizer': roberta_tokenizer},
        'bert': {'model': bert_model, 'tokenizer': bert_tokenizer},
        # 'deberta': {'model': deberta_model, 'tokenizer': deberta_tokenizer}
    }


def calculate_perplexity(text, model, tokenizer):
    """
    计算给定文本的困惑度（Perplexity）。
    """
    try:
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
        input_ids = encoded_input['input_ids']

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
            return perplexity.item()
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return None


def get_cls_embedding(text, model, tokenizer):
    """
    提取文本的 [CLS] token 嵌入，用作句子特征向量。
    """
    try:
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(**encoded_input)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            return cls_embedding.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

