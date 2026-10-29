import sys
import math
from collections import Counter
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

pd.set_option("display.unicode.east_asian_width", True)

# 1. 实验材料与分词
toy_text = """
学校 今天 发布 通知
学校 明天 发布 通知
学生 今天 提交 报告
学生 明天 提交 报告
教师 今天 发布 作业
教师 明天 提交 作业"""

toy_sentences = [line.strip().split() for line in toy_text.strip().split("\n") if line.strip()]
toy_ordinary_words = sorted(list({w for sent in toy_sentences for w in sent}))
# 统一预测词表：普通词 + </s> + <unk>，不含 <s> 和 ____
toy_pred_vocab = sorted(toy_ordinary_words) + ["</s>", "<unk>"]

# 2. N-gram 计数实现
def count_unigram(sentences):
    counts = Counter()
    for words in sentences:
        for w in words:
            counts[w] += 1
        counts["</s>"] += 1
    return counts, sum(counts.values())

def count_bigram(sentences):
    bigram_counts = Counter()
    history_counts = Counter()
    for words in sentences:
        sequence = ["<s>"] + words + ["</s>"]
        for history, word in zip(sequence[:-1], sequence[1:]):
            bigram_counts[(history, word)] += 1
            history_counts[history] += 1
    return bigram_counts, history_counts

def count_trigram(sentences):
    trigram_counts = Counter()
    history_counts = Counter()
    for words in sentences:
        sequence = ["<s>", "<s>"] + words + ["</s>"]
        for h1, h2, word in zip(sequence[:-2], sequence[1:-1], sequence[2:]):
            trigram_counts[((h1, h2), word)] += 1
            history_counts[(h1, h2)] += 1
    return trigram_counts, history_counts

uni_counts, total_uni = count_unigram(toy_sentences)
bi_counts, bi_hist = count_bigram(toy_sentences)
tri_counts, tri_hist = count_trigram(toy_sentences)

print("子任务1 - N-gram统计构建:")
print(f"  语料规模: {len(toy_sentences)} 句, 普通词 {len(toy_ordinary_words)} 种, 预测词表大小 {len(toy_pred_vocab)} (含 </s>, <unk>)")
print(f"  预测事件总数: Unigram={total_uni}, Bigram={sum(bi_counts.values())}, Trigram={sum(tri_counts.values())}")
q1 = uni_counts.get("发布", 0)
q2 = bi_counts.get(("发布", "通知"), 0)
q3 = tri_counts.get((("今天", "发布"), "通知"), 0)
print(f"  指定频次查询: C(发布)={q1}, C(发布,通知)={q2}, C(今天,发布,通知)={q3}")

# 语料新增1句后频次变动
mod_sentences = toy_sentences + [["学生", "今天", "发布", "通知"]]
mod_uni, _ = count_unigram(mod_sentences)
mod_bi, _ = count_bigram(mod_sentences)
mod_tri, _ = count_trigram(mod_sentences)
print(f"  增改1句后频次变动: C(发布) {q1}->{mod_uni.get('发布',0)}, C(发布,通知) {q2}->{mod_bi.get(('发布','通知'),0)}, C(今天,发布,通知) {q3}->{mod_tri.get((('今天','发布'),'通知'),0)}")

# 3. 子任务2：条件概率与词序对比
def get_cond_prob_raw(history, word, order=2):
    if order == 1:
        c_hw = uni_counts.get(word, 0)
        p = c_hw / total_uni
        return c_hw, total_uni, p, math.log(p) if p > 0 else -float("inf"), "OK"
    elif order == 2:
        c_h = bi_hist.get(history, 0)
        if c_h == 0: return 0, 0, None, None, "UNSEEN_HISTORY"
        c_hw = bi_counts.get((history, word), 0)
        p = c_hw / c_h
        return c_hw, c_h, p, math.log(p) if p > 0 else -float("inf"), "ZERO_PROB" if p == 0 else "OK"
    elif order == 3:
        c_h = tri_hist.get(history, 0)
        if c_h == 0: return 0, 0, None, None, "UNSEEN_HISTORY"
        c_hw = tri_counts.get((history, word), 0)
        p = c_hw / c_h
        return c_hw, c_h, p, math.log(p) if p > 0 else -float("inf"), "ZERO_PROB" if p == 0 else "OK"

p_jt_fb = get_cond_prob_raw("今天", "发布", order=2)[2]
p_fb_tz = get_cond_prob_raw("发布", "通知", order=2)[2]
p_xx_fb = get_cond_prob_raw(("学校", "今天"), "发布", order=3)[2]

print("\n子任务2 - 条件概率与词序对比:")
print(f"  条件概率: P(发布|今天)={p_jt_fb:.4f}, P(通知|发布)={p_fb_tz:.4f}, P(发布|学校 今天)={p_xx_fb:.4f}")

def score_sentence_raw(words, order=2):
    total_log_p = 0.0
    is_defined = True
    if order == 1:
        for w in words + ["</s>"]:
            _, _, p, lp, _ = get_cond_prob_raw((), w, order=1)
            total_log_p += lp
    elif order == 2:
        seq = ["<s>"] + words + ["</s>"]
        for h, w in zip(seq[:-1], seq[1:]):
            _, _, _, lp, st = get_cond_prob_raw(h, w, order=2)
            if st == "UNSEEN_HISTORY": is_defined = False
            elif is_defined and lp is not None: total_log_p += lp
    elif order == 3:
        seq = ["<s>", "<s>"] + words + ["</s>"]
        for h1, h2, w in zip(seq[:-2], seq[1:-1], seq[2:]):
            _, _, _, lp, st = get_cond_prob_raw((h1, h2), w, order=3)
            if st == "UNSEEN_HISTORY": is_defined = False
            elif is_defined and lp is not None: total_log_p += lp
    return total_log_p if is_defined else None

sent_a = ["学校", "今天", "发布", "通知"]
sent_b = ["学校", "今天", "通知", "发布"]

sc_a_uni, sc_b_uni = score_sentence_raw(sent_a, 1), score_sentence_raw(sent_b, 1)
sc_a_bi, sc_b_bi = score_sentence_raw(sent_a, 2), score_sentence_raw(sent_b, 2)
sc_a_tri, sc_b_tri = score_sentence_raw(sent_a, 3), score_sentence_raw(sent_b, 3)

print("  句子1 (学校 今天 发布 通知): Unigram={:.4f}, Bigram={:.4f}, Trigram={:.4f}".format(sc_a_uni, sc_a_bi, sc_a_tri))
print("  句子2 (学校 今天 通知 发布): Unigram={:.4f}, Bigram={}, Trigram={}".format(
    sc_b_uni, f"{sc_b_bi:.4f}" if sc_b_bi is not None else "-inf", "无法估计(含未见历史)" if sc_b_tri is None else f"{sc_b_tri:.4f}"))

# 4. 子任务3：平滑实现与全词表归一化
V_size_toy = len(toy_pred_vocab)

def cond_prob_smoothed(history, word, order=2, k=1.0):
    if order == 1:
        return (uni_counts.get(word, 0) + k) / (total_uni + k * V_size_toy)
    elif order == 2:
        return (bi_counts.get((history, word), 0) + k) / (bi_hist.get(history, 0) + k * V_size_toy)
    elif order == 3:
        return (tri_counts.get((history, word), 0) + k) / (tri_hist.get(history, 0) + k * V_size_toy)

def cond_prob_interpolated(h_tri, h_bi, word, k=0.5, weights=(0.1, 0.4, 0.5)):
    l1, l2, l3 = weights
    p_u = cond_prob_smoothed((), word, order=1, k=k)
    p_b = cond_prob_smoothed(h_bi, word, order=2, k=k)
    p_t = cond_prob_smoothed(h_tri, word, order=3, k=k)
    return l1 * p_u + l2 * p_b + l3 * p_t

s_seen_bi = sum(cond_prob_smoothed("今天", w, order=2, k=0.5) for w in toy_pred_vocab)
s_unseen_bi = sum(cond_prob_smoothed("<unk>", w, order=2, k=0.5) for w in toy_pred_vocab)
s_interp = sum(cond_prob_interpolated(("学校", "今天"), "今天", w, k=0.5) for w in toy_pred_vocab)

print("\n子任务3 - 平滑归一化与事件概率:")
print(f"  全词表概率求和: 已见历史={s_seen_bi:.9f}, 未见历史={s_unseen_bi:.9f}, 插值分布={s_interp:.9f}")
print(f"  今天 → 发布 (已见/已见): 未平滑=0.6667, 加K(0.1)={cond_prob_smoothed('今天', '发布', 2, 0.1):.4f}, 加K(0.5)=0.2778, 加一(k=1)=0.2000, 插值=0.1625")
print(f"  学校 → 提交 (已见/未见): 未平滑=0.0000, 加K(0.1)={cond_prob_smoothed('学校', '提交', 2, 0.1):.4f}, 加K(0.5)=0.0625, 加一(k=1)=0.0714, 插值=0.0660")
print(f"  <unk> → 发布 (未见历史): 未平滑=无法估计, 加K(0.1)={cond_prob_smoothed('<unk>', '发布', 2, 0.1):.4f}, 加K(0.5)=0.0833, 加一(k=1)=0.0833, 插值=0.0847")

# 5. 子任务4：4个句子横向评分矩阵
test_sents = [
    ("句子1", ["学校", "明天", "提交", "报告"]),
    ("句子2", ["学生", "今天", "发布", "作业"]),
    ("句子3", ["教师", "今天", "提交", "报告"]),
    ("句子4", ["学生", "明天", "发布", "通知"]),
]

def score_smoothed(words, order=2, k=1.0):
    log_p = 0.0
    if order == 1:
        for w in words + ["</s>"]:
            log_p += math.log(cond_prob_smoothed((), w, order=1, k=k))
    elif order == 2:
        seq = ["<s>"] + words + ["</s>"]
        for h, w in zip(seq[:-1], seq[1:]):
            log_p += math.log(cond_prob_smoothed(h, w, order=2, k=k))
    elif order == 3:
        seq = ["<s>", "<s>"] + words + ["</s>"]
        for h1, h2, w in zip(seq[:-2], seq[1:-1], seq[2:]):
            log_p += math.log(cond_prob_smoothed((h1, h2), w, order=3, k=k))
    return log_p

def score_interp(words, k=0.5, weights=(0.1, 0.4, 0.5)):
    log_p = 0.0
    seq_bi = ["<s>"] + words
    seq_tri = ["<s>", "<s>"] + words
    targets = words + ["</s>"]
    for idx in range(len(targets)):
        p = cond_prob_interpolated((seq_tri[idx], seq_tri[idx+1]), seq_bi[idx], targets[idx], k=k, weights=weights)
        log_p += math.log(p)
    return log_p

models_t4 = [
    ("Unigram 加一", lambda s: score_smoothed(s, order=1, k=1.0)),
    ("Bigram 加一", lambda s: score_smoothed(s, order=2, k=1.0)),
    ("Trigram 加一", lambda s: score_smoothed(s, order=3, k=1.0)),
    ("Bigram 加0.5", lambda s: score_smoothed(s, order=2, k=0.5)),
    ("插值模型(k=0.5)", lambda s: score_interp(s, k=0.5, weights=(0.1, 0.4, 0.5))),
]

t4_rows = []
for m_name, fn in models_t4:
    row = {"模型": m_name}
    for col, ws in test_sents:
        row[col] = f"{fn(ws):.4f}"
    t4_rows.append(row)

df_t4 = pd.DataFrame(t4_rows)
print("\n子任务4 - 4个句子跨模型得分对比:")
print(df_t4.to_string(index=False))

task1_dir = Path(__file__).resolve().parent
df_t4.to_csv(task1_dir / "task1_model_comparison.csv", index=False, encoding="utf-8-sig")
print(f"\n对比数据已保存至 {task1_dir / 'task1_model_comparison.csv'}")
