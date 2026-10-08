import sys
from collections import Counter
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 1. 准备小型语料
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
print("语料词数：", len(corpus))
print("前20个词：", corpus[:20])

# 2. 建立词表
word_counts = Counter(corpus)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}

print("\n词表大小：", len(vocab))
print("词表：", vocab)

task_dir = Path(__file__).resolve().parent

# 保存词表与词频分布 (支撑分析高低频词分布)
vocab_df = pd.DataFrame([{"词编号": word_to_id[w], "词语": w, "词频": word_counts[w]} for w in vocab])
vocab_df.to_csv(task_dir / "vocab.csv", index=False, encoding="utf-8-sig")

# 3. 构造 Skip-gram 训练样本
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

print("\n训练样本数：", len(pairs))
print("前10个训练样本：")
for center_id, context_id in pairs[:10]:
    print(id_to_word[center_id], "→", id_to_word[context_id])

# 4. 比较不同窗口大小并保存数据 (支撑回答思考题第1问)
print("\n不同窗口大小样本数：")
window_stats = []
for window_size in [1, 2, 3]:
    current_pairs = [pair for tokens in sentence_tokens
                     for pair in make_skipgram_pairs(tokens, word_to_id, window_size)]
    print(f"窗口大小={window_size}, 样本数={len(current_pairs)}")
    window_stats.append({"窗口大小": window_size, "样本数": len(current_pairs)})

pd.DataFrame(window_stats).to_csv(task_dir / "window_size_comparison.csv", index=False, encoding="utf-8-sig")
print(f"\n[已保存探索文件]\n- {task_dir / 'vocab.csv'}\n- {task_dir / 'window_size_comparison.csv'}")
