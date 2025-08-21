# Author: Hu Jia
# Date: 2025.8.21

import pandas as pd
import random
from tqdm import tqdm
import logging
import os

# 配置日志，用于在控制台显示进度
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def add_noise(text, noise_ratio=0.1, text_index=None):
    """
    随机向文本中插入无意义的字符。
    """
    logging.info(f"--- 正在对第 {text_index + 1} 条文本添加噪声 ---")
    noise_chars = ['#', '@', '$', '%', '&', '*', '!', '?']
    char_list = list(text)
    total_chars = len(char_list)
    num_noise = int(total_chars * noise_ratio)

    for _ in range(num_noise):
        insert_pos = random.randint(0, len(char_list))
        char_list.insert(insert_pos, random.choice(noise_chars))

    noisy_text = "".join(char_list)
    logging.info(f"加噪后文本长度 (字符数): {len(noisy_text)}")
    return noisy_text


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

    # --- 生成加噪数据集 ---
    logging.info("\n开始添加噪声...")
    df_noise = df_aigc.copy()
    df_noise['noisy_text'] = [add_noise(text, text_index=i)
                              for i, text in tqdm(enumerate(df_noise['text']), desc="加噪中")]

    # --- 合并并保存结果 ---
    df_final = df_aigc[['text', 'label']].rename(columns={'text': 'original_text', 'label': 'original_label'})
    df_final = df_final.join(df_noise['noisy_text'])

    output_file = 'noisy_text.csv'
    df_final.to_csv(output_file, index=False, encoding="utf-8-sig")
    logging.info(f"加噪后的数据已成功生成并保存到 {output_file}")


if __name__ == "__main__":
    main()
