# -*- coding: utf-8 -*-
"""
generate_aigc_texts.py
该脚本使用 gpt2-large 模型生成高质量、无干扰的AIGC文本，并保存为CSV文件。
"""
from transformers import pipeline, set_seed
import torch
import pandas as pd
import random
import re

# ========== 配置 ==========
NUM_SAMPLES = 10  # 总共要生成的AIGC文本数量
OUT_FILE = "aigc_texts.csv"
RANDOM_SEED = 42
PROMPTS = [
    # News Prompts (more diverse topics)
    "Write a detailed news report about a new scientific discovery in the field of astrophysics.",
    "Report on the economic impact of recent changes in global trade policy.",
    "Compose a news article about a major cultural event or festival taking place in a European city.",
    "Provide an update on the progress of a new space mission to Mars.",
    "Write a news story about the surprising outcome of a local election.",
    "Summarize the latest developments in sustainable urban planning.",

    # Academic Prompts (more diverse fields)
    "Compose an in-depth analysis of the ethical implications of using artificial intelligence in medical diagnosis.",
    "Summarize a research paper on the topic of quantum entanglement.",
    "Explain the principles of non-Euclidean geometry in a detailed, academic tone.",
    "Write a literature review on the use of symbolism in Victorian-era novels.",
    "Describe the process of photosynthesis and its importance to the ecosystem in a scientific report format.",
    "Provide a scholarly analysis of the socio-economic factors contributing to the decline of ancient empires.",

    # Novel/Creative Prompts (more diverse genres)
    "Continue a short story about a detective in a new city, focusing on the dark and moody atmosphere.",
    "Describe a journey through a fantastical, non-existent landscape, detailing the flora, fauna, and geography.",
    "Write the first chapter of a historical fiction novel set during the Renaissance.",
    "Compose a short story from the perspective of an animal observing human behavior.",
    "Narrate a science fiction tale about the first human contact with an alien civilization.",
    "Tell a mysterious story about an old, forgotten library where the books have a life of their own.",
    "Write a short piece of fantasy fiction centered on a magic-wielding blacksmith.",
]

# ========== 文本清理函数 ==========
def clean_text(text: str) -> str:
    """移除URL、引用信息和无关提示。"""
    # 去掉URL
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # 去掉括号中的年份/出版信息
    text = re.sub(r"\([^)]*\d{4}[^)]*\)", "", text)

    # 去掉“submit your article”之类的句子
    text = re.sub(r"submit your article.*", "", text, flags=re.IGNORECASE)

    # 去掉多余符号
    text = text.replace("  ", " ").strip()

    return text


def generate_and_clean_texts(num_samples):
    """
    使用预训练的语言模型生成文本，并进行严格的后处理。
    """
    print("--- 正在初始化AIGC模型... ---")
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline('text-generation', model='gpt2-large', device=device)
    set_seed(RANDOM_SEED)

    aigc_texts = []
    print(f"--- 正在生成 {num_samples} 条AIGC文本... ---")

    while len(aigc_texts) < num_samples:
        prompt = random.choice(PROMPTS)
        generation_prompt = f"{prompt}\nHere is the article:"

        try:
            generated = generator(
                generation_prompt,
                max_length=200,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.92,              # 更保守的nucleus采样
                top_k=50,               # 限制候选词
                repetition_penalty=1.5, # 强化惩罚，避免模式化输出
                pad_token_id=50256
            )

            text = generated[0]['generated_text']

            # 移除我们添加的引导句和原始prompt
            text = text.replace(generation_prompt, '').replace(prompt, '').strip()

            # 清理无关内容
            text = clean_text(text)

            # 强制移除多余空白
            text = ' '.join(text.split())

            if len(text.split()) > 30:  # 避免过短
                aigc_texts.append(text)

        except Exception as e:
            print(f"❌ 生成文本失败：{e}")

    return aigc_texts[:num_samples]


def main():
    """执行文本生成和保存操作。"""
    aigc_texts = generate_and_clean_texts(NUM_SAMPLES)

    # 将文本和标签 (1) 写入DataFrame
    data = pd.DataFrame({
        'text': aigc_texts,
        'label': [1] * len(aigc_texts)
    })

    data.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"\n✅ 成功将 {len(data)} 条AIGC文本保存到 {OUT_FILE}，格式为 (text, label)。")


if __name__ == "__main__":
    main()