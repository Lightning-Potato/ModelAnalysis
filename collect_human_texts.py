# -*- coding: utf-8 -*-
"""
collect_human_texts.py
该脚本用于收集指定数量的英文新闻、学术论文和小说文本，并保存为 CSV 文件，可用于后续数据分析。
"""
import requests
import csv
import random
from bs4 import BeautifulSoup
import time
import re
import feedparser  # 用于解析 arXiv API 的 XML 响应

# ========== 配置 ==========
# 总共收集文本数（注意该值应是3的倍数，因为分为3个类别）
NUM_TEXTS = 1602  # 您的目标总数
OUT_FILE = "human_texts.csv"
# 定义文本长度范围
MIN_LEN_CHARS_ACADEMIC = 200
MAX_LEN_CHARS_ACADEMIC = 2000
MIN_LEN_CHARS = 1250
MAX_LEN_CHARS = 2500


# ========== 新闻收集 (多RSS源) ==========
def fetch_full_article(url):
    """
    从给定的新闻文章 URL 中抓取并解析完整的文章正文。
    此函数尝试查找常见的文章主体标签（如 <article> 或特定 div）。
    """
    try:
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')

        # 尝试使用常见的选择器找到主要文章内容
        article_body = soup.find('article') or soup.find(class_='story-body') or soup.find(
            id='main-content') or soup.find(class_=re.compile(r'body'))

        if article_body:
            # 提取文本并清理多余的空白
            text = article_body.get_text(separator=' ', strip=True)
            return text
        else:
            return None
    except Exception as e:
        print(f"❌ 从 {url} 获取完整文章失败: {e}")
        return None


def fetch_news(num):
    """
    从多个新闻 RSS 源获取新闻文章链接，然后检索完整的文章内容。
    该函数在收集到目标数量的文章后停止。
    """
    print("--- 正在从多个RSS源获取新闻文本... ---")
    # 更新的新闻源列表，包含了您提供的网址
    rss_urls = [
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "http://rss.cnn.com/rss/edition_world.rss",
        "https://feeds.reuters.com/reuters/worldNews",
        "https://www.theguardian.com/world/rss",
    ]
    full_texts = []

    for url in rss_urls:
        if len(full_texts) >= num:
            break
        try:
            response = requests.get(url, timeout=10)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'xml')
            items = soup.find_all("item")

            for item in items:
                if len(full_texts) >= num:
                    break

                link = item.find("link").text if item.find("link") else None
                if link and "http" in link:
                    article_text = fetch_full_article(link)
                    if article_text and len(article_text) >= MIN_LEN_CHARS:
                        sentences = re.split(r'(?<=[.!?])\s+', article_text)
                        processed_text = ""
                        for sentence in sentences:
                            if len(processed_text + " " + sentence) <= MAX_LEN_CHARS:
                                processed_text += " " + sentence if processed_text else sentence
                            else:
                                break
                        if len(processed_text) >= MIN_LEN_CHARS:
                            full_texts.append(processed_text)
                            # print(f"✅ 已获取 {len(full_texts)}/{num} 篇新闻文本。")

                time.sleep(random.uniform(0.5, 2.0))

        except Exception as e:
            print(f"❌ 获取 RSS 源 {url} 失败: {e}")

    print(f"✅ 最终成功获取 {len(full_texts)} 篇完整新闻文本。")
    return full_texts


# ========== 学术论文 (从 arXiv 抓取摘要) ==========
def fetch_academic(num):
    """
    从 arXiv API 获取学术论文摘要。
    通过循环调用 API 并改变起始位置参数来获取更多结果，并从多个领域获取。
    """
    print("--- 正在从多个 arXiv 领域获取学术摘要 ---")
    texts = []
    start_index = 0
    max_results_per_call = 50  # 每次调用获取50篇摘要

    # 增加搜索领域以获取更多摘要
    search_queries = [
        "cat:cs.AI",  # 人工智能
        "cat:cs.CV",  # 计算机视觉
        "cat:math.CO",  # 组合数学
        "cat:stat.ML",  # 机器学习
        "cat:physics.ao-ph",  # 天体物理
        "cat:q-bio.GN",  # 基因组学
    ]

    for query in search_queries:
        if len(texts) >= num:
            break
        start_index = 0
        while len(texts) < num:
            url = f"http://export.arxiv.org/api/query?search_query=all:({query})&start={start_index}&max_results={max_results_per_call}"
            try:
                response = requests.get(url, timeout=15)
                feed = feedparser.parse(response.text)

                if not feed.entries:
                    print(f"⚠️ arXiv API 在 {query} 领域已无更多结果。")
                    break

                for entry in feed.entries:
                    if len(texts) >= num:
                        break

                    abstract = entry.summary.replace("\n", " ").strip()
                    if MIN_LEN_CHARS_ACADEMIC <= len(abstract) <= MAX_LEN_CHARS_ACADEMIC:
                        texts.append(abstract)

                # print(f"✅ 已获取 {len(texts)}/{num} 条学术摘要。")
                start_index += max_results_per_call
                time.sleep(random.uniform(0.3, 0.8))

            except Exception as e:
                print(f"❌ 从 arXiv API 获取摘要失败: {e}")
                break

    print(f"✅ 最终成功获取 {len(texts)} 条学术摘要。")
    return texts


# ========== 小说收集 (Project Gutenberg) ==========
def fetch_novels(num):
    """
    从 Project Gutenberg 获取小说文本。
    增加了大量URL以增加文本来源。
    """
    print("--- 正在获取小说文本... ---")
    urls = [
        # 经典文学作品
        "https://www.gutenberg.org/files/11/11-0.txt",  # Alice's Adventures in Wonderland
        "https://www.gutenberg.org/files/84/84-0.txt",  # Frankenstein
        "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
        "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick; Or, The Whale
        "https://www.gutenberg.org/files/43/43-0.txt",  # The Strange Case of Dr. Jekyll and Mr. Hyde
        "https://www.gutenberg.org/files/98/98-0.txt",  # A Tale of Two Cities
        "https://www.gutenberg.org/files/1661/1661-0.txt",  # The Adventures of Sherlock Holmes
        "https://www.gutenberg.org/files/2542/2542-0.txt",  # A Christmas Carol
        "https://www.gutenberg.org/files/42108/42108-0.txt",  # The Importance of Being Earnest
        "https://www.gutenberg.org/files/5200/5200-0.txt",  # Metamorphosis
        "https://www.gutenberg.org/files/1952/1952-0.txt",  # The Republic
        "https://www.gutenberg.org/files/64817/64817-0.txt",  # The Odyssey
        "https://www.gutenberg.org/files/76/76-0.txt",  # The Adventures of Tom Sawyer
        "https://www.gutenberg.org/files/1260/1260-0.txt",  # Jane Eyre
        "https://www.gutenberg.org/files/829/829-0.txt",  # The Rime of the Ancient Mariner
        "https://www.gutenberg.org/files/2600/2600-0.txt",  # War and Peace
        "https://www.gutenberg.org/files/2591/2591-0.txt",  # The Count of Monte Cristo
        "https://www.gutenberg.org/files/100/100-0.txt",  # The Complete Works of William Shakespeare
        "https://www.gutenberg.org/files/74/74-0.txt",  # The Story of My Life
        "https://www.gutenberg.org/files/19337/19337-0.txt",  # Twenty Thousand Leagues under the Sea
        "https://www.gutenberg.org/files/174/174-0.txt",  # The Picture of Dorian Gray
        "https://www.gutenberg.org/files/2852/2852-0.txt",  # The Divine Comedy
        "https://www.gutenberg.org/files/1184/1184-0.txt",  # The Scarlet Letter
        "https://www.gutenberg.org/files/19942/19942-0.txt",  # The Brothers Karamazov
        "https://www.gutenberg.org/files/844/844-0.txt",  # Don Quixote
        "https://www.gutenberg.org/files/23611/23611-0.txt",  # The Secret Garden
        "https://www.gutenberg.org/files/16478/16478-0.txt",  # The Jungle Book
        "https://www.gutenberg.org/files/244/244-0.txt",  # The Pilgrim's Progress
        "https://www.gutenberg.org/files/70364/70364-0.txt",  # A Study in Scarlet
        "https://www.gutenberg.org/files/345/345-0.txt",  # Dracula
        "https://www.gutenberg.org/files/209/209-0.txt",  # The Prince
        "https://www.gutenberg.org/files/21920/21920-0.txt",  # The Phantom of the Opera
        "https://www.gutenberg.org/files/1400/1400-0.txt",  # Great Expectations
        "https://www.gutenberg.org/files/2625/2625-0.txt",  # The Three Musketeers
        "https://www.gutenberg.org/files/3300/3300-0.txt",  # The Last of the Mohicans
        "https://www.gutenberg.org/files/540/540-0.txt",  # The Life and Adventures of Robinson Crusoe
        "https://www.gutenberg.org/files/25344/25344-0.txt",  # The Hound of the Baskervilles
        "https://www.gutenberg.org/files/1727/1727-0.txt",  # A Portrait of the Artist as a Young Man
        "https://www.gutenberg.org/files/125/125-0.txt",  # The Raven
        "https://www.gutenberg.org/files/105/105-0.txt",  # Utopia
        "https://www.gutenberg.org/files/209/209-0.txt",  # The Prince
        "https://www.gutenberg.org/files/11995/11995-0.txt",  # The Yellow Wallpaper
        "https://www.gutenberg.org/files/135/135-0.txt",  # The Red Badge of Courage
        "https://www.gutenberg.org/files/190/190-0.txt",  # The Age of Innocence
        "https://www.gutenberg.org/files/1123/1123-0.txt",  # The Adventures of Huckleberry Finn
        "https://www.gutenberg.org/files/145/145-0.txt",  # Middlemarch
        "https://www.gutenberg.org/files/1399/1399-0.txt",  # The House of the Seven Gables
        "https://www.gutenberg.org/files/1529/1529-0.txt",  # The Fall of the House of Usher
        "https://www.gutenberg.org/files/17709/17709-0.txt",  # The Little Prince
    ]
    texts = []
    for url in urls:
        if len(texts) >= num:
            break
        try:
            r = requests.get(url, timeout=15)
            r.encoding = 'utf-8'
            content = r.text
            parts = [p.replace("\r", " ").replace("\n", " ").strip() for p in content.split("\n\n") if
                     len(p.strip()) > 200]

            for part in parts:
                if len(texts) >= num:
                    break
                if MIN_LEN_CHARS <= len(part) <= MAX_LEN_CHARS:
                    texts.append(part)
            time.sleep(random.uniform(0.5, 2.0))
        except Exception as e:
            print(f"❌ 获取小说失败 ({url}): {e}")
    selected = texts[:num] if len(texts) >= num else texts
    print(f"✅ 已获取 {len(selected)} 条小说文本")
    return selected


# ========== 主函数 ==========
def main():
    """主函数，负责收集、处理和保存数据集"""
    all_texts = []
    num_per_category = NUM_TEXTS // 3

    # 新闻
    news_texts = fetch_news(num_per_category)
    news_filtered = [[text, 0] for text in news_texts]

    # 学术
    academic_texts = fetch_academic(num_per_category)
    academic_filtered = [[text, 0] for text in academic_texts]

    # 小说
    novel_texts = fetch_novels(num_per_category)
    novel_filtered = [[text, 0] for text in novel_texts]

    all_texts = news_filtered + academic_filtered + novel_filtered

    # 打印最终每个类别的数量
    print(f"\n--- 最终收集数量 ---")
    print(f"新闻文本: {len(news_filtered)} 条")
    print(f"学术文本: {len(academic_filtered)} 条")
    print(f"小说文本: {len(novel_filtered)} 条")

    # 打乱文本顺序，避免模型学习到数据排列规律
    random.shuffle(all_texts)

    # 保存 CSV（使用 utf-8-sig 编码以避免 Excel 打开乱码）
    with open(OUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(all_texts)

    print(f"\n✅ 成功将 {len(all_texts)} 条文本保存到 {OUT_FILE}")


if __name__ == "__main__":
    main()
