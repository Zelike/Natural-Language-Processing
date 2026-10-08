import sys
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

torch.manual_seed(42)

# 支持 Matplotlib 正常显示中文
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 语料与词表
sentences = [
    "猫 喜欢 吃 鱼",
    "狗 喜欢 吃 肉",
    "猫 和 狗 都 是 宠物",
    "小猫 是 可爱 的 宠物",
    "小狗 是 忠诚 的 宠物",
    "鱼 生活 在 水中",
    "鸟 喜欢 在 天空 飞翔",
    "飞机 在 天空 飞行",
]

corpus = " ".join(sentences).split()
word_counts = Counter(corpus)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
word_to_id = {word: i for i, word in enumerate(vocab)}

# 2. 提取词向量
task_dir = Path(__file__).resolve().parent
model_path = task_dir.parent / "task.3" / "skipgram_model.pt"

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center_words):
        return self.output(self.embedding(center_words))

embedding_dim = 10
model = SkipGram(len(vocab), embedding_dim=embedding_dim)

if model_path.exists():
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    def make_skipgram_pairs(tokens, w2i, window_size=1):
        pairs = []
        for center_pos, center_word in enumerate(tokens):
            c_id = w2i[center_word]
            left = max(0, center_pos - window_size)
            right = min(len(tokens), center_pos + window_size + 1)
            for context_pos in range(left, right):
                if context_pos != center_pos:
                    pairs.append((c_id, w2i[tokens[context_pos]]))
        return pairs
    
    sentence_tokens = [s.split() for s in sentences]
    pairs = [p for tokens in sentence_tokens for p in make_skipgram_pairs(tokens, word_to_id, 1)]
    centers = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    contexts = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    for _ in range(1000):
        optimizer.zero_grad()
        loss = criterion(model(centers), contexts)
        loss.backward()
        optimizer.step()

vectors = model.embedding.weight.detach().cpu().numpy()

# 3. PCA 降维至 2 维
pca = PCA(n_components=2)
points = pca.fit_transform(vectors)

# 4. 绘制并保存二维散点图
plt.figure(figsize=(9, 7))
plt.scatter(points[:, 0], points[:, 1], color="royalblue", edgecolors="black", s=70, alpha=0.8)

for i, word in enumerate(vocab):
    plt.text(points[i, 0] + 0.05, points[i, 1] + 0.05, word, fontsize=11)

plt.title("Word Embeddings Visualized by PCA", fontsize=14)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(alpha=0.3)

out_img = task_dir / "pca_visualization.png"
plt.savefig(out_img, dpi=300, bbox_inches="tight")
plt.close()

print(f"PCA 2维可视化散点图已生成并保存至: {out_img}")
