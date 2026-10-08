import sys
from collections import Counter
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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

# 2. 构造训练样本
def make_skipgram_pairs(tokens, word_to_id, window_size=1):
    pairs = []
    for center_pos, center_word in enumerate(tokens):
        center_id = word_to_id[center_word]
        left = max(0, center_pos - window_size)
        right = min(len(tokens), center_pos + window_size + 1)
        for context_pos in range(left, right):
            if context_pos == center_pos:
                continue
            context_word = tokens[context_pos]
            pairs.append((center_id, word_to_id[context_word]))
    return pairs

sentence_tokens = [sentence.split() for sentence in sentences]
pairs = [pair for tokens in sentence_tokens
         for pair in make_skipgram_pairs(tokens, word_to_id, window_size=1)]

centers = torch.tensor([p[0] for p in pairs], dtype=torch.long)
contexts = torch.tensor([p[1] for p in pairs], dtype=torch.long)

print("中心词张量形状：", centers.shape)
print("上下文词张量形状：", contexts.shape)

# 3. 定义 Skip-gram 模型
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center_words):
        vectors = self.embedding(center_words)
        logits = self.output(vectors)
        return logits

# 4. 训练模型
embedding_dim = 10
model = SkipGram(len(vocab), embedding_dim=embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.03)

loss_history = []

for epoch in range(1000):
    optimizer.zero_grad()
    logits = model(centers)
    loss = criterion(logits, contexts)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 200 == 0:
        print(f"epoch={epoch + 1}, loss={loss.item():.4f}")

# 5. 保存损失曲线与模型权重
task_dir = Path(__file__).resolve().parent

plt.figure(figsize=(7, 4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Skip-gram Training Loss")
plt.grid(alpha=0.3)
plt.savefig(task_dir / "loss_curve.png", dpi=300, bbox_inches="tight")
plt.close()

torch.save({
    "model_state_dict": model.state_dict(),
    "vocab": vocab,
    "word_to_id": word_to_id,
    "id_to_word": id_to_word,
    "embedding_dim": embedding_dim
}, task_dir / "skipgram_model.pt")

print(f"\n训练完成，损失曲线已保存至: {task_dir / 'loss_curve.png'}")
