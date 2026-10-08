import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

torch.manual_seed(42)

# 1. 语料与词表准备
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
id_to_word = {i: word for word, i in word_to_id.items()}

# 2. 获取训练好的词向量 (优先加载 task.3 权重，若无则快速训练)
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

embeddings = model.embedding.weight.detach()
print("词向量矩阵形状：", embeddings.shape)

# 3. 计算特定词对余弦相似度
def cosine_similarity(word_a, word_b):
    vec_a = embeddings[word_to_id[word_a]].unsqueeze(0)
    vec_b = embeddings[word_to_id[word_b]].unsqueeze(0)
    return F.cosine_similarity(vec_a, vec_b).item()

print("\n指定词对余弦相似度：")
test_pairs = [("猫", "狗"), ("天空", "飞机"), ("猫", "飞机"), ("鱼", "水中")]
for pair in test_pairs:
    print(f"{pair[0]} — {pair[1]}: {cosine_similarity(*pair):.4f}")

# 4. 查询最相似词
def most_similar(word, top_k=5):
    target_id = word_to_id[word]
    target = embeddings[target_id].unsqueeze(0)
    similarities = F.cosine_similarity(target, embeddings)

    result = []
    for idx in torch.argsort(similarities, descending=True):
        idx = idx.item()
        if idx == target_id:
            continue
        result.append((id_to_word[idx], float(similarities[idx])))
        if len(result) >= top_k:
            break
    return result

print("\n最相似词查询 (Top-5)：")
for word in ["猫", "狗", "天空", "宠物"]:
    sims = [f"{w}({s:.4f})" for w, s in most_similar(word)]
    print(f"{word} -> {', '.join(sims)}")

# 5. 回到任务1：对比真正训练得到的 Skip-gram 向量
def compare_representations(words, onehot_vectors, dense_vectors):
    records = []
    for i, j in combinations(range(len(words)), 2):
        row = {"词对": f"{words[i]}—{words[j]}"}
        for label, vectors in [("独热", onehot_vectors), ("分布式", dense_vectors)]:
            a, b = vectors[i], vectors[j]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            row[f"{label}_欧氏距离"] = float(np.linalg.norm(a - b))
            row[f"{label}_余弦相似度"] = float(a @ b / denom) if denom > 0 else np.nan
        records.append(row)
    return pd.DataFrame(records)

trained_compare_words = ["猫", "狗", "鸟", "飞机"]
trained_ids = [word_to_id[w] for w in trained_compare_words]
trained_vectors = embeddings[trained_ids].cpu().numpy()
trained_onehot = np.eye(len(vocab), dtype=float)[trained_ids]

trained_comparison = compare_representations(trained_compare_words, trained_onehot, trained_vectors)
print("\n训练后向量与独热表示指标对比表：")
print(trained_comparison.round(4).to_string(index=False))

trained_comparison.to_csv(task_dir / "task4_trained_comparison.csv", index=False, encoding="utf-8-sig")
print(f"\n对比表已导出至: {task_dir / 'task4_trained_comparison.csv'}")
