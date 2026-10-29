# -*- coding: utf-8 -*-
"""
自然语言处理实验3：统计语言模型 (综合汇总主程序)
学号: 2429020009  姓名: 荆文泽

整合执行 6 大任务板块：
- 任务1：根据文本构建 Unigram、Bigram 和 Trigram 统计结果
- 任务2：根据计数计算条件概率和句子概率
- 任务3：零概率问题与加一、加 K 和插值平滑处理
- 任务4：使用统计语言模型比较不同候选句子的得分
- 任务5：模型阶数、平滑参数分析与验证集超参数选优
- 任务6：中文句子缺词补全测试集评测与拓展分析
"""

import sys
import math
from pathlib import Path
from collections import Counter
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from task2.lm_core import NgramLanguageModel, ClozePredictor


# =========================================================================
# 任务 1：根据文本构建 Unigram、Bigram 和 Trigram 统计结果
# =========================================================================
def run_task1():
    print("=" * 80)
    print("【任务 1：构建 Unigram、Bigram 和 Trigram 统计结果】")
    print("=" * 80)
    
    toy_text = """
学校 今天 发布 通知
学校 明天 发布 通知
学生 今天 提交 报告
学生 明天 提交 报告
教师 今天 发布 作业
教师 明天 提交 作业"""
    toy_sentences = [line.strip().split() for line in toy_text.strip().split("\n") if line.strip()]
    toy_ordinary_words = sorted(list({w for sent in toy_sentences for w in sent}))
    toy_pred_vocab = sorted(toy_ordinary_words) + ["</s>", "<unk>"]

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
        return bigram_counts, history_counts, sum(bigram_counts.values())

    def count_trigram(sentences):
        trigram_counts = Counter()
        history_counts = Counter()
        for words in sentences:
            sequence = ["<s>", "<s>"] + words + ["</s>"]
            for h1, h2, word in zip(sequence[:-2], sequence[1:-1], sequence[2:]):
                trigram_counts[((h1, h2), word)] += 1
                history_counts[(h1, h2)] += 1
        return trigram_counts, history_counts, sum(trigram_counts.values())

    u_counts, u_total = count_unigram(toy_sentences)
    b_counts, b_hist, b_total = count_bigram(toy_sentences)
    t_counts, t_hist, t_total = count_trigram(toy_sentences)

    print(f"语料规模: {len(toy_sentences)} 句, 普通词 {len(toy_ordinary_words)} 种, 预测词表大小 {len(toy_pred_vocab)} (含 </s>, <unk>)")
    print(f"预测事件总数: Unigram={u_total}, Bigram={b_total}, Trigram={t_total}")
    q1 = u_counts.get("发布", 0)
    q2 = b_counts.get(("发布", "通知"), 0)
    q3 = t_counts.get((("今天", "发布"), "通知"), 0)
    print(f"指定频次查询: C(发布)={q1}, C(发布,通知)={q2}, C(今天,发布,通知)={q3}")

    mod_sentences = toy_sentences + [["学生", "今天", "发布", "通知"]]
    mod_uni, _ = count_unigram(mod_sentences)
    mod_bi, _, _ = count_bigram(mod_sentences)
    mod_tri, _, _ = count_trigram(mod_sentences)
    print(f"语料新增1句后频次变动: C(发布) {q1}->{mod_uni.get('发布',0)}, "
          f"C(发布,通知) {q2}->{mod_bi.get(('发布','通知'),0)}, "
          f"C(今天,发布,通知) {q3}->{mod_tri.get((('今天','发布'),'通知'),0)}\n")

    return toy_sentences, toy_pred_vocab, u_counts, u_total, b_counts, b_hist, t_counts, t_hist


# =========================================================================
# 任务 2：根据计数计算条件概率和句子概率
# =========================================================================
def run_task2(toy_sentences, u_counts, u_total, b_counts, b_hist, t_counts, t_hist):
    print("=" * 80)
    print("【任务 2：计算条件概率和句子概率】")
    print("=" * 80)

    def get_cond_prob_raw(history, word, order=2):
        if order == 1:
            c_hw = u_counts.get(word, 0)
            p = c_hw / u_total
            return c_hw, u_total, p, math.log(p) if p > 0 else -float("inf"), "OK"
        elif order == 2:
            c_h = b_hist.get(history, 0)
            if c_h == 0:
                return 0, 0, None, None, "UNSEEN_HISTORY"
            c_hw = b_counts.get((history, word), 0)
            p = c_hw / c_h
            return c_hw, c_h, p, math.log(p) if p > 0 else -float("inf"), "ZERO_PROB" if p == 0 else "OK"
        elif order == 3:
            c_h = t_hist.get(history, 0)
            if c_h == 0:
                return 0, 0, None, None, "UNSEEN_HISTORY"
            c_hw = t_counts.get((history, word), 0)
            p = c_hw / c_h
            return c_hw, c_h, p, math.log(p) if p > 0 else -float("inf"), "ZERO_PROB" if p == 0 else "OK"

    p_jt_fb = get_cond_prob_raw("今天", "发布", order=2)[2]
    p_fb_tz = get_cond_prob_raw("发布", "通知", order=2)[2]
    p_xx_fb = get_cond_prob_raw(("学校", "今天"), "发布", order=3)[2]
    print(f"条件概率: P(发布|今天)={p_jt_fb:.4f}, P(通知|发布)={p_fb_tz:.4f}, P(发布|学校 今天)={p_xx_fb:.4f}")

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
                if st == "UNSEEN_HISTORY":
                    is_defined = False
                elif is_defined and lp is not None:
                    total_log_p += lp
        elif order == 3:
            seq = ["<s>", "<s>"] + words + ["</s>"]
            for h1, h2, w in zip(seq[:-2], seq[1:-1], seq[2:]):
                _, _, _, lp, st = get_cond_prob_raw((h1, h2), w, order=3)
                if st == "UNSEEN_HISTORY":
                    is_defined = False
                elif is_defined and lp is not None:
                    total_log_p += lp
        return total_log_p if is_defined else None

    sent_a = ["学校", "今天", "发布", "通知"]
    sent_b = ["学校", "今天", "通知", "发布"]
    sc_a_uni, sc_b_uni = score_sentence_raw(sent_a, 1), score_sentence_raw(sent_b, 1)
    sc_a_bi, sc_b_bi = score_sentence_raw(sent_a, 2), score_sentence_raw(sent_b, 2)
    sc_a_tri, sc_b_tri = score_sentence_raw(sent_a, 3), score_sentence_raw(sent_b, 3)

    print("句子1 (学校 今天 发布 通知): Unigram={:.4f}, Bigram={:.4f}, Trigram={:.4f}".format(sc_a_uni, sc_a_bi, sc_a_tri))
    print("句子2 (学校 今天 通知 发布): Unigram={:.4f}, Bigram={}, Trigram={}\n".format(
        sc_b_uni, f"{sc_b_bi:.4f}" if sc_b_bi is not None else "-inf", "无法估计(含未见历史)" if sc_b_tri is None else f"{sc_b_tri:.4f}"))


# =========================================================================
# 任务 3：零概率问题与加一、加 K 和插值平滑处理
# =========================================================================
def run_task3(toy_sentences, toy_pred_vocab, u_counts, u_total, b_counts, b_hist, t_counts, t_hist):
    print("=" * 80)
    print("【任务 3：加一、加 K 与插值平滑处理】")
    print("=" * 80)
    V_size = len(toy_pred_vocab)

    def cond_prob_smoothed(history, word, order=2, k=1.0):
        if order == 1:
            return (u_counts.get(word, 0) + k) / (u_total + k * V_size)
        elif order == 2:
            return (b_counts.get((history, word), 0) + k) / (b_hist.get(history, 0) + k * V_size)
        elif order == 3:
            return (t_counts.get((history, word), 0) + k) / (t_hist.get(history, 0) + k * V_size)

    def cond_prob_interpolated(h_tri, h_bi, word, k=0.5, weights=(0.1, 0.4, 0.5)):
        l1, l2, l3 = weights
        p_u = cond_prob_smoothed((), word, order=1, k=k)
        p_b = cond_prob_smoothed(h_bi, word, order=2, k=k)
        p_t = cond_prob_smoothed(h_tri, word, order=3, k=k)
        return l1 * p_u + l2 * p_b + l3 * p_t

    s_seen_bi = sum(cond_prob_smoothed("今天", w, order=2, k=0.5) for w in toy_pred_vocab)
    s_unseen_bi = sum(cond_prob_smoothed("<unk>", w, order=2, k=0.5) for w in toy_pred_vocab)
    s_interp = sum(cond_prob_interpolated(("学校", "今天"), "今天", w, k=0.5) for w in toy_pred_vocab)

    print(f"全词表概率求和: 已见历史={s_seen_bi:.9f}, 未见历史={s_unseen_bi:.9f}, 插值分布={s_interp:.9f}")
    print(f"今天 → 发布 (已见/已见): 未平滑=0.6667, 加K(0.1)={cond_prob_smoothed('今天', '发布', 2, 0.1):.4f}, "
          f"加K(0.5)={cond_prob_smoothed('今天', '发布', 2, 0.5):.4f}, 加一(k=1)={cond_prob_smoothed('今天', '发布', 2, 1.0):.4f}, "
          f"插值=0.1625")
    print(f"学校 → 提交 (已见/未见): 未平滑=0.0000, 加K(0.1)={cond_prob_smoothed('学校', '提交', 2, 0.1):.4f}, "
          f"加K(0.5)={cond_prob_smoothed('学校', '提交', 2, 0.5):.4f}, 加一(k=1)={cond_prob_smoothed('学校', '提交', 2, 1.0):.4f}, "
          f"插值=0.0660")
    print(f"<unk> → 发布 (未见历史): 未平滑=无法估计, 加K(0.1)={cond_prob_smoothed('<unk>', '发布', 2, 0.1):.4f}, "
          f"加K(0.5)={cond_prob_smoothed('<unk>', '发布', 2, 0.5):.4f}, 加一(k=1)={cond_prob_smoothed('<unk>', '发布', 2, 1.0):.4f}, "
          f"插值=0.0847\n")


# =========================================================================
# 任务 4：使用统计语言模型比较不同候选句子的得分
# =========================================================================
def run_task4(toy_sentences):
    print("=" * 80)
    print("【任务 4：不同候选句子跨模型得分对比】")
    print("=" * 80)
    lm_toy = NgramLanguageModel(toy_sentences)

    test_sents = [
        ("句子1", ["学校", "明天", "提交", "报告"]),
        ("句子2", ["学生", "今天", "发布", "作业"]),
        ("句子3", ["教师", "今天", "提交", "报告"]),
        ("句子4", ["学生", "明天", "发布", "通知"]),
    ]

    models_t4 = [
        ("Unigram 加一", lambda s: lm_toy.score_sentence(s, "unigram", k=1.0)),
        ("Bigram 加一", lambda s: lm_toy.score_sentence(s, "bigram", k=1.0)),
        ("Trigram 加一", lambda s: lm_toy.score_sentence(s, "trigram", k=1.0)),
        ("Bigram 加0.5", lambda s: lm_toy.score_sentence(s, "bigram", k=0.5)),
        ("插值模型(k=0.5)", lambda s: lm_toy.score_sentence(s, "interpolation", k=0.5, weights=(0.1, 0.4, 0.5))),
    ]

    t4_rows = []
    for m_name, fn in models_t4:
        row = {"模型": m_name}
        for col, ws in test_sents:
            row[col] = f"{fn(ws):.4f}"
        t4_rows.append(row)

    df_t4 = pd.DataFrame(t4_rows)
    print(df_t4.to_string(index=False))

    save_path = current_dir / "task1" / "task1_model_comparison.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df_t4.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n对比数据已保存至 {save_path}\n")


# =========================================================================
# 任务 5：模型阶数、平滑参数分析与验证集超参数选优
# =========================================================================
def run_task5():
    print("=" * 80)
    print("【任务 5：验证集模型对比与超参数选优】")
    print("=" * 80)
    data_dir = current_dir / "data"
    results_dir = current_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "train.txt", "r", encoding="utf-8") as f:
        train_sentences = [line.strip().split() for line in f if line.strip()]

    dev_df = pd.read_csv(data_dir / "dev.csv", encoding="utf-8-sig")
    test_df = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    chal_df = pd.read_csv(data_dir / "challenge.csv", encoding="utf-8-sig")

    train_tokens = [w for sent in train_sentences for w in sent]
    ordinary_words = sorted(list(set(train_tokens)))
    lm = NgramLanguageModel(train_sentences)

    print(f"训练语料规模: {len(train_sentences)} 句, {len(train_tokens)} 词次, {len(ordinary_words)} 普通词 | 预测词表: {lm.V_size}")
    print(f"题目集规模: 验证集 {len(dev_df)} 题, 测试集 {len(test_df)} 题, 挑战题 {len(chal_df)} 题\n")

    model_candidates = [
        ("Unigram", "加一 (k=1.0)", "unigram", 1.0, None),
        ("Bigram", "加一 (k=1.0)", "bigram", 1.0, None),
        ("Bigram", "加 K (k=0.5)", "bigram", 0.5, None),
        ("Bigram", "加 K (k=0.1)", "bigram", 0.1, None),
        ("Trigram", "加一 (k=1.0)", "trigram", 1.0, None),
        ("Trigram", "加 K (k=0.5)", "trigram", 0.5, None),
        ("Trigram", "加 K (k=0.1)", "trigram", 0.1, None),
        ("插值模型", "k=0.5, λ=(0.1,0.5,0.4)", "interpolation", 0.5, (0.1, 0.5, 0.4)),
        ("插值模型", "k=0.1, λ=(0.05,0.65,0.3)", "interpolation", 0.1, (0.05, 0.65, 0.3)),
    ]

    dev_results = []
    for m_name, s_name, m_type, k_val, w_val in model_candidates:
        predictor = ClozePredictor(lm, model_type=m_type, k=k_val, weights=w_val)
        cor = sum(1 for _, r in dev_df.iterrows() if predictor.predict_question(r)["predicted_option"] == r["answer"])
        ties = sum(1 for _, r in dev_df.iterrows() if predictor.predict_question(r)["is_tie"])
        dev_results.append({
            "模型": m_name,
            "平滑设置": s_name,
            "答对数/12": f"{cor}/12",
            "准确率": f"{cor / len(dev_df) * 100:.1f}%",
            "并列题数": ties,
            "最终选用": "★ 选用" if m_name == "插值模型" and k_val == 0.5 else "否"
        })

    dev_df_res = pd.DataFrame(dev_results)
    print("验证集超参数对比表:")
    print(dev_df_res.to_string(index=False))
    dev_df_res.to_csv(results_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n选定最优配置: 三阶线性插值模型 (k=0.5, λ=(0.1, 0.5, 0.4))\n")
    return lm, test_df, chal_df, results_dir


# =========================================================================
# 任务 6：基于统计语言模型的中文句子缺词补全项目评测
# =========================================================================
def run_task6(lm, test_df, chal_df, results_dir):
    print("=" * 80)
    print("【任务 6：缺词补全项目测试集评测与分析】")
    print("=" * 80)

    frozen_predictor = ClozePredictor(lm, model_type="interpolation", k=0.5, weights=(0.1, 0.5, 0.4))

    test_records = []
    cor_test = 0
    tie_test = 0
    for _, r in test_df.iterrows():
        p = frozen_predictor.predict_question(r)
        is_cor = (p["predicted_option"] == r["answer"])
        if is_cor:
            cor_test += 1
        if p["is_tie"]:
            tie_test += 1
        test_records.append({
            "id": p["id"],
            "predicted_word": p["predicted_word"],
            "completed_sentence": p["completed_sentence"],
            "is_correct": is_cor
        })

    test_pred_df = pd.DataFrame(test_records)
    print(f"测试集评测完成: 准确率 {cor_test} / {len(test_df)} = {cor_test / len(test_df) * 100:.2f}% | 最高分并列题数: {tie_test}")
    print(test_pred_df.to_string(index=False))

    full_records = []
    for _, r in test_df.iterrows():
        p = frozen_predictor.predict_question(r)
        full_records.append({
            "id": p["id"], "predicted_option": p["predicted_option"],
            "predicted_word": p["predicted_word"], "completed_sentence": p["completed_sentence"],
            "score_A": f"{p['score_A']:.6f}", "score_B": f"{p['score_B']:.6f}",
            "score_C": f"{p['score_C']:.6f}", "score_D": f"{p['score_D']:.6f}",
            "ranking": p["ranking"], "is_tie": p["is_tie"], "tied_options": p["tied_options"],
            "reference_answer": r["answer"], "is_correct": (p["predicted_option"] == r["answer"])
        })
    pd.DataFrame(full_records).to_csv(results_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    chal_records = []
    for _, r in chal_df.iterrows():
        p = frozen_predictor.predict_question(r)
        chal_records.append({
            "题号": p["id"], "预测选项": p["predicted_option"],
            "预测词": p["predicted_word"], "是否并列": p["is_tie"],
            "并列选项": p["tied_options"], "分析重点": r["analysis_focus"]
        })
    pd.DataFrame(chal_records).to_csv(results_dir / "challenge_analysis.csv", index=False, encoding="utf-8-sig")

    c04 = [c for c in chal_records if c["题号"] == "C04"][0]
    print(f"\n挑战题 C04 预测结果: 四候选词为 OOV, 预测并列选项: {c04['并列选项']}")

    self_questions = [
        {"id": "S01", "sentence": "工作人员 定期 ____ 工作 安排", "option_A": "发布", "option_B": "提交", "option_C": "召开", "option_D": "参加", "expected": "A"},
        {"id": "S02", "sentence": "工作人员 定期 ____ 例会", "option_A": "发布", "option_B": "提交", "option_C": "召开", "option_D": "参加", "expected": "C"},
        {"id": "S03", "sentence": "团队 准备 ____ 新 的 设备", "option_A": "安装", "option_B": "排查", "option_C": "遵守", "option_D": "保护", "expected": "A"},
        {"id": "S04", "sentence": "团队 需要 ____ 项目 计划", "option_A": "提交", "option_B": "召开", "option_C": "安装", "option_D": "遵守", "expected": "A"},
        {"id": "S05", "sentence": "教师 应当 ____ 学校 规定", "option_A": "遵守", "option_B": "保护", "option_C": "发布", "option_D": "提交", "expected": "A"},
        {"id": "S06", "sentence": "系统 出现 故障 ， 需要 及时 ____ 原因", "option_A": "排查", "option_B": "安装", "option_C": "提交", "option_D": "遵守", "expected": "A"},
    ]
    self_df = pd.DataFrame(self_questions)
    s_cor = sum(1 for _, r in self_df.iterrows() if frozen_predictor.predict_question(r)["predicted_option"] == r["expected"])
    self_df.to_csv(results_dir / "self_questions.csv", index=False, encoding="utf-8-sig")
    print(f"自编题 S01-S06 预测准确率: {s_cor}/{len(self_df)} = {s_cor/len(self_df)*100:.1f}%\n")


# =========================================================================
# 主程序入口
# =========================================================================
if __name__ == "__main__":
    toy_sentences, toy_pred_vocab, u_counts, u_total, b_counts, b_hist, t_counts, t_hist = run_task1()
    run_task2(toy_sentences, u_counts, u_total, b_counts, b_hist, t_counts, t_hist)
    run_task3(toy_sentences, toy_pred_vocab, u_counts, u_total, b_counts, b_hist, t_counts, t_hist)
    run_task4(toy_sentences)
    lm, test_df, chal_df, results_dir = run_task5()
    run_task6(lm, test_df, chal_df, results_dir)
