import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import random
from tqdm import tqdm
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义模型路径
TRANSLATOR_ZH_EN = "Helsinki-NLP/opus-mt-zh-en"  # 中文到英文
TRANSLATOR_EN_ZH = "Helsinki-NLP/opus-mt-en-zh"  # 英文到中文

# 模型缓存
models_cache = {}


def initialize_models():
    """初始化并加载所需的模型和分词器"""
    if 'translator' not in models_cache:
        try:
            logging.info("正在加载翻译模型...")
            zh_en_tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_ZH_EN)
            zh_en_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR_ZH_EN)
            en_zh_tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_EN_ZH)
            en_zh_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR_EN_ZH)
            models_cache['translator'] = (zh_en_tokenizer, zh_en_model, en_zh_tokenizer, en_zh_model)
        except Exception as e:
            logging.error(f"加载翻译模型失败: {e}")
            return None

    return models_cache['translator']


def translate_en_zh_en(text, zh_en_tokenizer, zh_en_model, en_zh_tokenizer, en_zh_model, text_index):
    """
    将英文 -> 中文 -> 英文
    """
    try:
        logging.info(f"--- 正在对第 {text_index + 1} 条文本进行回译 (英文→中文→英文) ---")

        # 英文到中文
        zh_inputs = en_zh_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        zh_outputs = en_zh_model.generate(**zh_inputs)
        zh_text = en_zh_tokenizer.decode(zh_outputs[0], skip_special_tokens=True)

        # 中文到英文
        en_inputs = zh_en_tokenizer(zh_text, return_tensors="pt", max_length=512, truncation=True)
        en_outputs = zh_en_model.generate(**en_inputs)
        final_text = zh_en_tokenizer.decode(en_outputs[0], skip_special_tokens=True)

        logging.info(f"回译后文本长度 (字符数): {len(final_text)}")
        return final_text
    except Exception as e:
        logging.error(f"对第 {text_index + 1} 条文本回译失败: {e}")
        return None


def main():
    input_file = 'aigc_texts.csv'
    if not os.path.exists(input_file):
        logging.error(f"错误: 文件 {input_file} 未找到。")
        return

    logging.info("正在加载原始数据集...")
    # ✅ 指定 utf-8-sig 读入，避免乱码
    df_original = pd.read_csv(input_file, encoding="utf-8-sig")

    if not {"text", "label"}.issubset(df_original.columns):
        logging.error("CSV 文件必须包含 'text' 和 'label' 两个表头！")
        return

    # 只处理 AIGC 文本
    df_aigc = df_original[df_original['label'] == 1].copy().reset_index(drop=True)
    if df_aigc.empty:
        logging.warning("数据集中没有 AIGC 文本 (label == 1)，无法生成测试数据。")
        return

    df_aigc = df_aigc.head(2)  # 仅取前 5 条用于测试

    # 初始化模型
    models = initialize_models()
    if models is None:
        return
    zh_en_tokenizer, zh_en_model, en_zh_tokenizer, en_zh_model = models

    # --- 生成英文→中文→英文 回译数据 ---
    logging.info("\n开始英文→中文→英文 回译...")
    df_translation = df_aigc.copy()
    df_translation['translated_text'] = [
        translate_en_zh_en(text, zh_en_tokenizer, zh_en_model, en_zh_tokenizer, en_zh_model, i)
        for i, text in tqdm(enumerate(df_translation['text']), desc="翻译中")
    ]

    # --- 合并结果 ---
    logging.info("\n正在合并结果...")
    df_final = df_aigc[['text', 'label']].rename(columns={'text': 'original_text', 'label': 'original_label'})
    df_final = df_final.join(df_translation['translated_text'])

    # ✅ 保存时也用 utf-8-sig，避免乱码
    output_file = 'translated_text.csv'
    df_final.to_csv(output_file, index=False, encoding="utf-8-sig")
    logging.info(f"所有数据已生成并保存到 {output_file}")


if __name__ == "__main__":
    main()
