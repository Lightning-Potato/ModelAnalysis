import pandas as pd
import chardet

# 输入文件
HUMAN_FILE = "human_texts.csv"
AIGC_FILE = "aigc_texts.csv"
FINAL_FILE = "my_dataset.csv"

def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, "rb") as f:
        raw_data = f.read(50000)  # 取前 50KB 样本
    result = chardet.detect(raw_data)
    return result["encoding"]

def load_and_fix(file_path, encoding, label):
    """读取CSV并转码，添加标签"""
    try:
        df = pd.read_csv(file_path, encoding=encoding, on_bad_lines="skip")
    except Exception as e:
        print(f"❌ 读取 {file_path} 失败: {e}")
        return pd.DataFrame(columns=["text", "label"])

    # 确保有 text 列
    if "text" not in df.columns:
        df.columns = ["text"]

    # 转为字符串再转码，防止乱码
    df["text"] = df["text"].astype(str).apply(
        lambda x: x.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    )

    # 添加标签
    df["label"] = label
    return df[["text", "label"]]

if __name__ == "__main__":
    # 检测文件编码
    human_encoding = detect_encoding(HUMAN_FILE)
    aigc_encoding = detect_encoding(AIGC_FILE)

    print(f"检测到 {HUMAN_FILE} 的编码为：{human_encoding}")
    print(f"检测到 {AIGC_FILE} 的编码为：{aigc_encoding}")

    # 读取并修复
    human_df = load_and_fix(HUMAN_FILE, human_encoding, 0)
    aigc_df = load_and_fix(AIGC_FILE, aigc_encoding, 1)

    # 合并
    print("--- 正在合并数据集... ---")
    combined_df = pd.concat([human_df, aigc_df], ignore_index=True)

    print(f"人类文本数量：{len(human_df)}")
    print(f"AIGC文本数量：{len(aigc_df)}")
    print(f"合并前总计：{len(combined_df)} 条样本")

    # ✅ 打乱数据
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"✅ 最终数据集已创建并打乱，总计 {len(combined_df)} 条样本。")

    # 保存为 UTF-8 带 BOM，避免 Excel 乱码
    combined_df.to_csv(FINAL_FILE, index=False, encoding="utf-8-sig")
    print(f"文件已保存为 {FINAL_FILE}")
