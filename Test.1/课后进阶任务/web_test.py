import os
import requests
from bs4 import BeautifulSoup
import jieba
from collections import Counter

# 1. 目标 URL：提示词工程核心要素
url = "https://www.promptingguide.ai/zh/introduction/elements"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

print(f"[*] 正在抓取大模型提示词要素页面: {url} ...")
response = requests.get(url, headers=headers, timeout=10)
response.encoding = "utf-8"  # 强制使用 utf-8 解码中文

# 2. 精准定位正文区（<article> 标签）
soup = BeautifulSoup(response.text, "html.parser")
article = soup.find("article") or soup.find("main")

if not article:
    print("[!] 未能定位到 article 标签，采用备选提取")
    raw_text = soup.get_text()
else:
    raw_text = article.get_text()

print(f"[+] 抓取成功！解析出的指南正文字符数: {len(raw_text)}")

# 3. 领域特定词库增强（大模型领域专属专有名词）
domain_words = ["提示词", "提示工程", "语言模型", "输入数据", "输出指示", "上下文", "大模型"]
for w in domain_words:
    jieba.add_word(w)

# 4. jieba 分词与停用词清洗
words = jieba.lcut(raw_text)

# 过滤无意义符号与单字虚词，保留关键概念
stopwords = {"一个", "下面", "为了", "相关", "一些", "可以", "以及", "通过", "进行"}
clean_words = [
    w.strip() for w in words 
    if len(w.strip()) > 1 and w.strip() not in stopwords
]

# 5. 统计并展示前 10 个核心高频词
word_counts = Counter(clean_words)
print("\n" + "=" * 45)
print("【《提示词要素》页面核心高频关键词 Top 10】")
print("=" * 45)
for rank, (word, freq) in enumerate(word_counts.most_common(10), 1):
    print(f"Top {rank:02d}: {word:<12} -> 出现 {freq} 次")

# 6. 将前 10 结果同步保存至文本文件
current_dir = os.path.dirname(os.path.abspath(__file__))
result_file = os.path.join(current_dir, "web_test_result.txt")

with open(result_file, "w", encoding="utf-8") as f:
    f.write(f"目标网页: {url}\n")
    f.write(f"解析字符总数: {len(raw_text)}\n\n")
    f.write("【《提示词要素》页面核心高频关键词 Top 10】\n")
    f.write("-" * 45 + "\n")
    for rank, (word, freq) in enumerate(word_counts.most_common(10), 1):
        f.write(f"Top {rank:02d}: {word:<12} -> 出现 {freq} 次\n")

print(f"\n[+] 实验结果已同步保存至: {result_file}")