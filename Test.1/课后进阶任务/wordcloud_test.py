import os
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 配置中文字体，防止标题乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 动态加载 content.txt 文本
current_dir = os.path.dirname(os.path.abspath(__file__))
txt_path = os.path.join(current_dir, "wordcloud_content.txt")
# 如果在当前子目录没找到，则向上查找父级目录
if not os.path.exists(txt_path):
    txt_path = os.path.join(current_dir, "..", "content.txt")

with open(txt_path, "r", encoding="utf-8") as f:
    text = f.read()

print(f"[*] 成功从 {os.path.basename(txt_path)} 读取到 {len(text)} 个字符")

# 2. 用 jieba 分词并清洗单字虚词
words = jieba.lcut(text)
clean_words = [w.strip() for w in words if len(w.strip()) > 1]
text_cut = " ".join(clean_words)

# 3. 生成词云
wc = WordCloud(
    font_path="C:/Windows/Fonts/msyh.ttc",
    background_color="white",
    width=800,
    height=600,
    max_words=50
).generate(text_cut)

# 4. 显示并保存
plt.figure(figsize=(10, 6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("NLP 实验：中文词云展示", fontsize=16)

save_path = os.path.join(current_dir, "wordcloud_result.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"[+] 词云图已更新并保存至: {save_path}")

plt.show()