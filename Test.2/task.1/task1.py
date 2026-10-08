import sys
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

pd.set_option("display.unicode.east_asian_width", True)

# 1. 独热表示
demo_words = ["猫", "狗", "鸟", "飞机"]
demo_ids = {w: i for i, w in enumerate(demo_words)}
demo_onehot = np.eye(len(demo_words), dtype=float)

print("独热表示：")
print(pd.DataFrame(demo_onehot, index=demo_words, columns=demo_words))

# 2. 从语料统计上下文共现
demo_sentences = [
    "可爱 猫 宠物", "可爱 狗 宠物",
    "家中 猫 陪伴", "家中 狗 陪伴",
    "猫 吃 鱼", "狗 吃 肉",
    "鸟 天空 飞翔", "鸟 树林 飞翔",
    "飞机 天空 飞行", "飞机 机场 飞行",
    "鸟 空中 移动", "飞机 空中 移动",
]

def build_cooccurrence(sentences, words, word_ids):
    tokens = [s.split() for s in sentences]
    contexts = sorted({w for sent in tokens for w in sent} - set(words))
    context_ids = {w: i for i, w in enumerate(contexts)}
    counts = np.zeros((len(words), len(contexts)), dtype=float)
    for sent in tokens:
        for pos, word in enumerate(sent):
            if word not in word_ids:
                continue
            for j in range(max(0, pos - 2), min(len(sent), pos + 3)):
                if j != pos and sent[j] in context_ids:
                    counts[word_ids[word], context_ids[sent[j]]] += 1
    return counts, contexts

demo_counts, demo_contexts = build_cooccurrence(demo_sentences, demo_words, demo_ids)
print("\n共现矩阵：")
print(pd.DataFrame(demo_counts, index=demo_words, columns=demo_contexts).to_string())

# 3. 截断 SVD 获取 3 维分布式表示
def get_svd_dense(counts):
    x = counts / np.maximum(np.linalg.norm(counts, axis=1, keepdims=True), 1e-12)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    dense = u[:, :3] * s[:3]
    return dense

demo_dense = get_svd_dense(demo_counts)
assert demo_dense.shape == (4, 3)
assert np.all(np.linalg.norm(demo_dense, axis=1) > 0)

print("\n分布式表示 (SVD 3维)：")
print(pd.DataFrame(demo_dense, index=demo_words, columns=["维度1", "维度2", "维度3"]).round(4))

# 4. 指标对比函数
def compare_representations(words, onehot_vectors, dense_vectors):
    records = []
    for i, j in combinations(range(len(words)), 2):
        row = {"词对": f"{words[i]}—{words[j]}"}
        for label, vectors in [("独热", onehot_vectors), ("分布式", dense_vectors)]:
            a, b = vectors[i], vectors[j]
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            row[f"{label}_欧氏距离"] = float(np.linalg.norm(a - b))
            row[f"{label}_余弦相似度"] = float(a @ b / denominator) if denominator > 0 else np.nan
        records.append(row)
    df = pd.DataFrame(records)
    for col in df.columns:
        if col != "词对":
            df[col] = df[col].apply(lambda x: f"{x:.4f}")
    return df

demo_comparison = compare_representations(demo_words, demo_onehot, demo_dense)
print("\n指标对比表：")
print(demo_comparison.to_string(index=False))

# 5. 余弦相似度矩阵与导出
def row_cosine_matrix(vectors):
    normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return normalized @ normalized.T

print("\n独热表示余弦相似度矩阵：")
print(pd.DataFrame(row_cosine_matrix(demo_onehot), index=demo_words, columns=demo_words).round(4))
print("\n分布式表示余弦相似度矩阵：")
print(pd.DataFrame(row_cosine_matrix(demo_dense), index=demo_words, columns=demo_words).round(4))

task_dir = Path(__file__).resolve().parent
demo_comparison.to_csv(task_dir / "task1_comparison.csv", index=False, encoding="utf-8-sig")

# 6. 语料扰动对比 (修改“狗”短语为飞行短语)
modified_sentences = demo_sentences.copy()
modified_sentences[3] = "狗 天空 飞行"  # 替换原有的 "家中 狗 陪伴"

mod_counts, _ = build_cooccurrence(modified_sentences, demo_words, demo_ids)
mod_dense = get_svd_dense(mod_counts)
mod_comparison = compare_representations(demo_words, demo_onehot, mod_dense)
mod_comparison.to_csv(task_dir / "task1_modified_comparison.csv", index=False, encoding="utf-8-sig")

cat_dog_orig = demo_comparison.loc[demo_comparison["词对"] == "猫—狗", "分布式_余弦相似度"].values[0]
cat_dog_mod = mod_comparison.loc[mod_comparison["词对"] == "猫—狗", "分布式_余弦相似度"].values[0]
print(f"\n语料修改后 猫—狗 相似度变化: {cat_dog_orig} -> {cat_dog_mod}")
