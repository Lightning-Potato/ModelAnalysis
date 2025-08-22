# Author: Hu Jia
# Date: 2025.8.22

import pandas as pd
import os
import logging

# 配置日志以便更好地反馈
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_label_data(file_path, label):
    """
    读取CSV文件，尝试多种常见编码，并添加标签列。

    参数:
        file_path (str): CSV文件的路径。
        label (int): 要分配给文本的标签（人类为0，AIGC为1）。

    返回:
        pd.DataFrame: 包含“text”和“label”列的DataFrame。如果文件无法读取，则返回空的DataFrame。
    """
    if not os.path.exists(file_path):
        logging.error(f"错误：文件 '{file_path}' 未找到。")
        return pd.DataFrame(columns=["text", "label"])

    encodings_to_try = ['utf-8', 'utf-8-sig', 'gbk', 'latin1']
    df = pd.DataFrame()

    for encoding in encodings_to_try:
        try:
            logging.info(f"正在尝试使用 '{encoding}' 编码读取文件：{file_path}...")
            df = pd.read_csv(file_path, encoding=encoding, on_bad_lines="skip")
            logging.info(f"成功使用 '{encoding}' 编码。")

            if "text" not in df.columns and len(df.columns) > 0:
                # 如果没有'text'列头，则假设第一列为文本
                df = df.iloc[:, [0]].rename(columns={df.columns[0]: "text"})

            break
        except Exception as e:
            logging.warning(f"使用 '{encoding}' 编码读取失败: {e}")
            continue

    if df.empty or "text" not in df.columns:
        logging.error(f"无法正确读取文件 '{file_path}'。请检查文件格式。")
        return pd.DataFrame(columns=["text", "label"])

    df["label"] = label
    # 确保文本列是字符串类型，并移除空值/NaN值
    df['text'] = df['text'].astype(str).str.strip()
    df.dropna(subset=['text'], inplace=True)
    df = df[df['text'] != '']
    return df[["text", "label"]]


def combine_datasets(human_file, aigc_file, output_file):
    """
    合并人类文本数据集和修改后的AIGC文本数据集，打乱数据，并保存结果。

    参数:
        human_file (str): 人类文本CSV文件的路径。
        aigc_file (str): 修改后的AIGC文本CSV文件的路径。
        output_file (str): 保存合并后数据集的路径。
    """
    logging.info(f"--- 正在开始合并 {os.path.basename(human_file)} 和 {os.path.basename(aigc_file)} ---")

    human_df = load_and_label_data(human_file, 0)
    aigc_df = load_and_label_data(aigc_file, 1)

    if human_df.empty or aigc_df.empty:
        logging.error("一个或两个输入文件都无法加载。合并已中止。")
        return

    combined_df = pd.concat([human_df, aigc_df], ignore_index=True)

    logging.info(f"已加载人类文本样本数: {len(human_df)}")
    logging.info(f"已加载AIGC文本样本数: {len(aigc_df)}")
    logging.info(f"打乱前总样本数: {len(combined_df)}")

    # 彻底打乱合并后的数据集，以实现均衡的评估
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    logging.info(f"✅ 最终数据集已创建并打乱，总样本数: {len(combined_df)}")

    # 将最终的DataFrame保存为CSV文件。
    # 使用 utf-8-sig 编码以确保与Excel等应用程序的兼容性
    try:
        combined_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        logging.info(f"文件成功保存到 {output_file}")
    except Exception as e:
        logging.error(f"保存输出文件失败: {e}")


if __name__ == "__main__":
    HUMAN_FILE = "human_texts.csv"

    # --- 每个鲁棒性测试的示例用法 ---

    # # 示例 1：合并人类文本和改写文本
    # REWRITE_FILE = "rewritten_texts.csv"
    # OUTPUT_REWRITE_FILE = "robustness_rewritten.csv"
    # combine_datasets(HUMAN_FILE, REWRITE_FILE, OUTPUT_REWRITE_FILE)

    # # 示例 2：合并人类文本和翻译文本
    # TRANSLATED_FILE = "translated_texts.csv"
    # OUTPUT_TRANSLATED_FILE = "robustness_translated.csv"
    # combine_datasets(HUMAN_FILE, TRANSLATED_FILE, OUTPUT_TRANSLATED_FILE)

    # 示例 3：合并人类文本和噪声文本
    NOISY_FILE = "noisy_texts.csv"
    OUTPUT_NOISY_FILE = "robustness_noisy.csv"
    combine_datasets(HUMAN_FILE, NOISY_FILE, OUTPUT_NOISY_FILE)
