import sys
from collections import Counter
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
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

# 2. 构造 CBOW 训练样本 (上下文词列表 -> 中心词)
def make_cbow_pairs(tokens, word_to_id, window_size=1):
    pairs = []
    for center_pos, center_word in enumerate(tokens):
        center_id = word_to_id[center_word]
        left = max(0, center_pos - window_size)
        right = min(len(tokens), center_pos + window_size + 1)

        context_ids = [word_to_id[tokens[i]] for i in range(left, right) if i != center_pos]
        if context_ids:
            pairs.append((context_ids, center_id))
    return pairs

sentence_tokens = [sentence.split() for sentence in sentences]
cbow_pairs = [pair for tokens in sentence_tokens
              for pair in make_cbow_pairs(tokens, word_to_id, window_size=1)]

print(f"CBOW (window_size=1) 训练样本数: {len(cbow_pairs)}")
print("前5个 CBOW 样本：")
for contexts, center in cbow_pairs[:5]:
    ctx_words = [id_to_word[c] for c in contexts]
    print(f"上下文: {ctx_words} → 中心词: {id_to_word[center]}")

# 3. 定义 CBOW 模型 (多上下文向量均值聚合)
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_indices_list):
        batch_vectors = []
        for ctx in context_indices_list:
            ctx_tensor = torch.tensor(ctx, dtype=torch.long)
            vec = self.embedding(ctx_tensor).mean(dim=0)
            batch_vectors.append(vec)
        stacked = torch.stack(batch_vectors)
        logits = self.output(stacked)
        return logits

# 4. 训练 CBOW 模型
embedding_dim = 10
cbow_model = CBOW(len(vocab), embedding_dim=embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cbow_model.parameters(), lr=0.03)

all_contexts = [p[0] for p in cbow_pairs]
all_targets = torch.tensor([p[1] for p in cbow_pairs], dtype=torch.long)

loss_history = []

for epoch in range(1000):
    optimizer.zero_grad()
    logits = cbow_model(all_contexts)
    loss = criterion(logits, all_targets)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 200 == 0:
        print(f"epoch={epoch + 1}, loss={loss.item():.4f}")

# 5. 提取向量与相似度计算
cbow_embeddings = cbow_model.embedding.weight.detach()

def cbow_cosine(word_a, word_b):
    va = cbow_embeddings[word_to_id[word_a]].unsqueeze(0)
    vb = cbow_embeddings[word_to_id[word_b]].unsqueeze(0)
    return F.cosine_similarity(va, vb).item()

test_pairs = [("猫", "狗"), ("天空", "飞机"), ("猫", "飞机"), ("鱼", "水中")]
print("\nCBOW 训练后词对余弦相似度：")
for pair in test_pairs:
    print(f"{pair[0]} — {pair[1]}: {cbow_cosine(*pair):.4f}")

# 6. 保存损失曲线与横向对比表
task_dir = Path(__file__).resolve().parent

# 保存损失曲线
plt.figure(figsize=(7, 4))
plt.plot(loss_history, color="darkorange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CBOW Training Loss")
plt.grid(alpha=0.3)
plt.savefig(task_dir / "cbow_loss_curve.png", dpi=300, bbox_inches="tight")
plt.close()

# 横向对比 Skip-gram (来自 task.3) 与 CBOW
skipgram_path = task_dir.parent / "task.3" / "skipgram_model.pt"
comp_records = []
if skipgram_path.exists():
    sg_data = torch.load(skipgram_path)
    sg_emb = sg_data["model_state_dict"]["embedding.weight"]
    for w1, w2 in test_pairs:
        v1 = sg_emb[word_to_id[w1]].unsqueeze(0)
        v2 = sg_emb[word_to_id[w2]].unsqueeze(0)
        sg_sim = F.cosine_similarity(v1, v2).item()
        cb_sim = cbow_cosine(w1, w2)
        comp_records.append({
            "词对": f"{w1}—{w2}",
            "Skip-gram余弦相似度": round(sg_sim, 4),
            "CBOW余弦相似度": round(cb_sim, 4)
        })
    pd.DataFrame(comp_records).to_csv(task_dir / "cbow_vs_skipgram_comparison.csv", index=False, encoding="utf-8-sig")

print(f"\n[已保存探索文件]\n- {task_dir / 'cbow_loss_curve.png'}\n- {task_dir / 'cbow_vs_skipgram_comparison.csv'}")
