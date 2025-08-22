# 作者: 胡嘉
# 日期: 2025.8.22
# 脚本功能: 这是一个修改后的脚本，用于根据指定的列名合并数据集。

import pandas as pd
import os
import logging

# 配置日志以便更好地反馈
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_label_data(file_path, text_column_name, label):
    """
    读取CSV文件，尝试多种常见编码，并根据指定的列名提取文本并添加标签列。

    参数:
        file_path (str): CSV文件的路径。
        text_column_name (str): 包含文本数据的列的名称。
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
            # 尝试带文件头读取，因为你的文件有表头
            df = pd.read_csv(file_path, encoding=encoding, on_bad_lines="skip")

            if text_column_name not in df.columns:
                logging.warning(f"警告: 文件 '{file_path}' 中找不到指定的列 '{text_column_name}'。跳过此编码。")
                continue

            logging.info(f"成功使用 '{encoding}' 编码。")

            # 将指定的文本列重命名为'text'，并删除其他列
            df = df[[text_column_name]].rename(columns={text_column_name: "text"})
            break
        except Exception as e:
            logging.warning(f"使用 '{encoding}' 编码读取失败: {e}")
            continue

    if df.empty or "text" not in df.columns:
        logging.error(f"无法正确读取文件 '{file_path}' 或找不到指定的列。请检查文件内容和列名。")
        return pd.DataFrame(columns=["text", "label"])

    # 为所有文本添加标签
    df["label"] = label
    # 确保文本列是字符串类型，并移除空值/NaN值
    df['text'] = df['text'].astype(str).str.strip()
    df.dropna(subset=['text'], inplace=True)
    df = df[df['text'] != '']
    return df[["text", "label"]]


def combine_datasets(human_file, aigc_file, aigc_text_column, output_file):
    """
    合并人类文本数据集和修改后的AIGC文本数据集，打乱数据，并保存结果。

    参数:
        human_file (str): 人类文本CSV文件的路径。
        aigc_file (str): 修改后的AIGC文本CSV文件的路径。
        aigc_text_column (str): AIGC文件中的文本列名。
        output_file (str): 保存合并后数据集的路径。
    """
    logging.info(f"--- 正在开始合并 {os.path.basename(human_file)} 和 {os.path.basename(aigc_file)} ---")

    # 指定 human_texts.csv 的文本列名是 'text'，标签为 0
    human_df = load_and_label_data(human_file, 'text', 0)
    # 指定 aigc 文件的文本列名，标签为 1
    aigc_df = load_and_label_data(aigc_file, aigc_text_column, 1)

    if human_df.empty or aigc_df.empty:
        logging.error("一个或两个输入文件都无法加载。合并已中止。")
        return

    # 诊断性输出：检查加载的数据是否正确
    logging.info("\n--- 诊断: 检查加载的数据头 ---")
    logging.info(f"人类文本数据头:\n{human_df.head()}")
    logging.info(f"AIGC文本数据头:\n{aigc_df.head()}")
    logging.info("----------------------------------\n")

    # 合并两个DataFrame
    combined_df = pd.concat([human_df, aigc_df], ignore_index=True)

    # 清除重复的文本行（如果存在的话）
    initial_count = len(combined_df)
    combined_df.drop_duplicates(subset=['text'], inplace=True)
    deduplicated_count = len(combined_df)
    if initial_count > deduplicated_count:
        logging.warning(f"警告：发现了 {initial_count - deduplicated_count} 条重复文本，已移除。")

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

    # --- 请选择一个要运行的鲁棒性测试，取消其注释并运行脚本 ---

    # --- 1. 合并人类文本和改写文本（rewritten） ---
    # REWRITTEN_FILE = "rewritten_texts.csv"
    # # 请根据你的 rewritten_texts.csv 文件头，指定正确的文本列名
    # REWRITTEN_TEXT_COLUMN = "rewritten_text"
    # OUTPUT_REWRITTEN_FILE = "robustness_rewritten.csv"
    # combine_datasets(HUMAN_FILE, REWRITTEN_FILE, REWRITTEN_TEXT_COLUMN, OUTPUT_REWRITTEN_FILE)

    # # --- 2. 合并人类文本和翻译文本（translated） ---
    # TRANSLATED_FILE = "translated_texts.csv"
    # # 请根据你的 translated_texts.csv 文件头，指定正确的文本列名
    # TRANSLATED_TEXT_COLUMN = "translated_text"
    # OUTPUT_TRANSLATED_FILE = "robustness_translated.csv"
    # combine_datasets(HUMAN_FILE, TRANSLATED_FILE, TRANSLATED_TEXT_COLUMN, OUTPUT_TRANSLATED_FILE)

    # --- 3. 合并人类文本和噪声文本（noisy） ---
    NOISY_FILE = "noisy_texts.csv"
    # 请根据你的 noisy_texts.csv 文件头，指定正确的文本列名
    NOISY_TEXT_COLUMN = "noisy_text"
    OUTPUT_NOISY_FILE = "robustness_noisy.csv"
    combine_datasets(HUMAN_FILE, NOISY_FILE, NOISY_TEXT_COLUMN, OUTPUT_NOISY_FILE)
