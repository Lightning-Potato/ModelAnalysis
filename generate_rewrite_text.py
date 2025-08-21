# Author: Hu Jia
# Date: 2025.8.21

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import random
from tqdm import tqdm
import logging
import os

# 配置日志，用于在控制台显示进度
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义模型路径
REWRITER_MODEL = "google/flan-t5-base"  # 用于文本改写

# 全局缓存，用于存储已加载的模型和分词器
models_cache = {}


def initialize_rewriter_model():
    """
    初始化并加载文本改写模型和分词器。
    """
    if 'rewriter' not in models_cache:
        try:
            logging.info("正在加载文本改写模型...")
            rewriter_tokenizer = AutoTokenizer.from_pretrained(REWRITER_MODEL)
            rewriter_model = AutoModelForSeq2SeqLM.from_pretrained(REWRITER_MODEL)
            models_cache['rewriter'] = (rewriter_tokenizer, rewriter_model)
        except Exception as e:
            logging.error(f"加载文本改写模型失败: {e}")
            return None

    return models_cache['rewriter']


def split_text_into_chunks(text, max_length=450, overlap=100):
    """
    将长文本分割成重叠的块，以适应模型输入限制。
    """
    tokens = text.split()
    chunks = []
    if len(tokens) <= max_length:
        return [text]

    start = 0
    while start < len(tokens):
        end = min(start + max_length, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start += (max_length - overlap)

    return chunks


def rewrite_text(text, tokenizer, model, text_index):
    """
    使用 Flan-T5 模型改写文本，支持长文本处理。
    """
    try:
        logging.info(f"--- 正在对第 {text_index + 1} 条文本进行改写 ---")
        chunks = split_text_into_chunks(text, max_length=450, overlap=100)
        rewritten_chunks = []

        for i, chunk in enumerate(chunks):
            prompt = f"Paraphrase the following text: {chunk}"
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                temperature=0.7
            )
            rewritten_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if rewritten_text.startswith("Paraphrased text:"):
                rewritten_text = rewritten_text.replace("Paraphrased text:", "").strip()
            rewritten_chunks.append(rewritten_text)

        final_rewritten_text = " ".join(rewritten_chunks)
        logging.info(f"最终改写后文本长度 (字符数): {len(final_rewritten_text)}")
        return final_rewritten_text
    except Exception as e:
        logging.error(f"对第 {text_index + 1} 条文本进行改写失败: {e}")
        return None


def main():
    input_file = 'aigc_texts.csv'
    if not os.path.exists(input_file):
        logging.error(f"错误: 文件 {input_file} 未找到。请确保文件存在。")
        return

    logging.info("正在加载原始数据集...")
    df_original = pd.read_csv(input_file, encoding="utf-8-sig")

    if not {"text", "label"}.issubset(df_original.columns):
        logging.error("CSV 文件必须包含 'text' 和 'label' 两个表头！")
        return

    # 筛选出 label == 1 的AIGC文本进行处理
    df_aigc = df_original[df_original['label'] == 1].copy().reset_index(drop=True)
    if df_aigc.empty:
        logging.warning("数据集中没有 'AIGC文本' (label == 1)，无法生成测试数据。")
        return

    # 仅取前 5 条用于测试
    df_aigc = df_aigc.head(5)

    # 初始化模型
    models = initialize_rewriter_model()
    if models is None:
        return
    rewriter_tokenizer, rewriter_model = models

    # --- 生成改写数据集 ---
    logging.info("\n开始文本改写...")
    df_rewrite = df_aigc.copy()
    df_rewrite['rewritten_text'] = [rewrite_text(text, rewriter_tokenizer, rewriter_model, i)
                                    for i, text in tqdm(enumerate(df_rewrite['text']), desc="改写中")]

    # --- 合并并保存结果 ---
    df_final = df_aigc[['text', 'label']].rename(columns={'text': 'original_text', 'label': 'original_label'})
    df_final = df_final.join(df_rewrite['rewritten_text'])

    output_file = 'rewritten_text.csv'
    df_final.to_csv(output_file, index=False, encoding="utf-8-sig")
    logging.info(f"改写后的数据已成功生成并保存到 {output_file}")


if __name__ == "__main__":
    main()
