# -*- coding: utf-8 -*-
"""
collect_human_texts.py
修复版：收集英文新闻、学术文本、小说，保存为 CSV，可直接用于数据分析。
"""
import requests
import csv
import random
from bs4 import BeautifulSoup
import time

# ========== 配置 ==========
NUM_TEXTS = 1602  # 总共收集文本数（注意整除 3）
OUT_FILE = "human_texts.csv"

# ========== 新闻收集 (BBC RSS) ==========
def fetch_news(num):
    print("--- 正在获取新闻文本... ---")
    url = "http://feeds.bbci.co.uk/news/rss.xml"
    try:
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'xml')
        items = soup.find_all("item")
        texts = [item.title.text + " " + item.description.text for item in items if item.description]
        print(f"✅ 已获取 {len(texts[:num])} 条新闻文本")
        return texts[:num]
    except Exception as e:
        print(f"❌ 获取新闻失败: {e}")
        return []

# ========== 学术收集 (arXiv API) ==========
def fetch_academic(num):
    print("--- 正在获取学术文本... ---")
    url = f"http://export.arxiv.org/api/query?search_query=cat:cs.AI&start=0&max_results={num}"
    try:
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, "xml")
        abstracts = [entry.summary.text.strip().replace("\n", " ") for entry in soup.find_all("entry")]
        print(f"✅ 已获取 {len(abstracts[:num])} 条学术文本")
        return abstracts[:num]
    except Exception as e:
        print(f"❌ 获取学术文本失败: {e}")
        return []

# ========== 小说收集 (Project Gutenberg) ==========
def fetch_novels(num):
    print("--- 正在获取小说文本... ---")
    urls = [
        "https://www.gutenberg.org/files/11/11-0.txt",        # Alice in Wonderland
        "https://www.gutenberg.org/files/84/84-0.txt",        # Frankenstein
        "https://www.gutenberg.org/files/1342/1342-0.txt",    # Pride and Prejudice
        "https://www.gutenberg.org/files/2701/2701-0.txt",    # Moby Dick
        "https://www.gutenberg.org/files/43/43-0.txt",        # Sherlock Holmes
        "https://www.gutenberg.org/files/98/98-0.txt",        # A Tale of Two Cities
        "https://www.gutenberg.org/files/1661/1661-0.txt",    # Tom Sawyer
        "https://www.gutenberg.org/files/2542/2542-0.txt",    # Crime and Punishment
        "https://www.gutenberg.org/files/42108/42108-0.txt", # The Great Gatsby
        "https://www.gutenberg.org/files/5200/5200-0.txt",    # Metamorphosis
        "https://www.gutenberg.org/files/1952/1952-0.txt",    # The Yellow Wallpaper
        "https://www.gutenberg.org/files/64817/64817-0.txt",  # Dorian Gray
        "https://www.gutenberg.org/files/76/76-0.txt",        # Huckleberry Finn
        "https://www.gutenberg.org/files/1260/1260-0.txt",    # Wuthering Heights
        "https://www.gutenberg.org/files/829/829-0.txt",      # The Brothers Karamazov
    ]
    texts = []
    for url in urls:
        try:
            r = requests.get(url, timeout=15)
            r.encoding = 'utf-8'
            content = r.text
            # 按段落切分
            parts = [p.replace("\r", " ").replace("\n", " ").strip() for p in content.split("\n\n") if len(p.strip())>200]
            # 取随机样本，控制每段 500~1000 字
            for part in parts:
                if 500 <= len(part) <= 1000:
                    texts.append(part)
            time.sleep(1)
        except Exception as e:
            print(f"❌ 获取小说失败 ({url}): {e}")
    selected = texts[:num] if len(texts) >= num else texts
    print(f"✅ 已获取 {len(selected)} 条小说文本")
    return selected

# ========== 主函数 ==========
def main():
    all_texts = []
    num_per_category = NUM_TEXTS // 3

    # 新闻
    news_texts = fetch_news(num_per_category)
    for text in news_texts:
        all_texts.append([text[:1000], 0])

    # 学术
    academic_texts = fetch_academic(num_per_category)
    for text in academic_texts:
        all_texts.append([text[:1000], 0])

    # 小说
    novel_texts = fetch_novels(num_per_category)
    for text in novel_texts:
        all_texts.append([text[:1000], 0])

    # 保存 CSV（utf-8-sig 避免 Excel 打开乱码）
    with open(OUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(all_texts)

    print(f"\n✅ 成功将 {len(all_texts)} 条文本保存到 {OUT_FILE}")

if __name__ == "__main__":
    main()
