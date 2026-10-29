import sys
import math
from pathlib import Path
import pandas as pd

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
from lm_core import NgramLanguageModel, ClozePredictor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

pd.set_option("display.unicode.east_asian_width", True)

data_dir = current_dir.parent / "data"
results_dir = current_dir.parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# 1. 子任务1：数据读取与结构统计
with open(data_dir / "train.txt", "r", encoding="utf-8") as f:
    train_sentences = [line.strip().split() for line in f if line.strip()]

train_tokens = [w for sent in train_sentences for w in sent]
ordinary_words = sorted(list(set(train_tokens)))
pred_vocab = sorted(ordinary_words) + ["</s>", "<unk>"]

dev_df = pd.read_csv(data_dir / "dev.csv", encoding="utf-8-sig")
test_df = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
chal_df = pd.read_csv(data_dir / "challenge.csv", encoding="utf-8-sig")

print("子任务1 - 数据集统计:")
print(f"  训练集: {len(train_sentences)} 句, {len(train_tokens)} 词次, {len(ordinary_words)} 普通词 | 预测词表大小: {len(pred_vocab)}")
print(f"  题目规模: 验证集 {len(dev_df)} 题, 测试集 {len(test_df)} 题, 挑战题 {len(chal_df)} 题")

# 2. 子任务2 & 3：模型建立与补全样例推断
lm = NgramLanguageModel(train_sentences)
demo_predictor = ClozePredictor(lm, model_type="bigram", k=1.0)
demo_res = demo_predictor.predict_question(dev_df.iloc[0])
print("\n子任务2 & 3 - 语言模型与缺词补全样例推断:")
print(f"  模型初始化完成: 词表槽位 {lm.V_size} | D01样例补全: 预测 {demo_res['predicted_option']} ({demo_res['predicted_word']}), 是否并列: {demo_res['is_tie']}")

# 3. 子任务4：验证集参数对比与模型选优
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
        "模型": m_name, "平滑设置": s_name, "答对数/12": f"{cor}/12",
        "准确率": f"{cor / len(dev_df) * 100:.1f}%", "并列题数": ties,
        "最终选用": "★ 选用" if m_name == "插值模型" and k_val == 0.5 else "否"
    })

dev_df_res = pd.DataFrame(dev_results)
print("\n子任务4 - 验证集选参对比结果:")
print(dev_df_res.to_string(index=False))
dev_df_res.to_csv(results_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")

FROZEN_MODEL, FROZEN_K, FROZEN_WEIGHTS = "interpolation", 0.5, (0.1, 0.5, 0.4)
print(f"  选定最优模型: 三阶线性插值模型 (k={FROZEN_K}, λ={FROZEN_WEIGHTS})")

# 4. 子任务5：测试集评测
frozen_predictor = ClozePredictor(lm, model_type=FROZEN_MODEL, k=FROZEN_K, weights=FROZEN_WEIGHTS)

test_records = []
cor_test = 0
tie_test = 0
for _, r in test_df.iterrows():
    p = frozen_predictor.predict_question(r)
    is_cor = (p["predicted_option"] == r["answer"])
    if is_cor: cor_test += 1
    if p["is_tie"]: tie_test += 1
    test_records.append({
        "id": p["id"], "predicted_option": p["predicted_option"],
        "predicted_word": p["predicted_word"], "completed_sentence": p["completed_sentence"],
        "score_A": f"{p['score_A']:.6f}", "score_B": f"{p['score_B']:.6f}",
        "score_C": f"{p['score_C']:.6f}", "score_D": f"{p['score_D']:.6f}",
        "ranking": p["ranking"], "is_tie": p["is_tie"], "tied_options": p["tied_options"],
        "reference_answer": r["answer"], "is_correct": is_cor
    })

test_pred_df = pd.DataFrame(test_records)
test_pred_df.to_csv(results_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

print(f"\n子任务5 - 测试集评测:")
print(f"  准确率: {cor_test} / {len(test_df)} = {cor_test / len(test_df) * 100:.2f}% | 最高分并列题数: {tie_test}")
print(f"  预测结果明细已保存至 {results_dir / 'test_predictions.csv'}")

# 5. 子任务6：挑战题与自编题评测
chal_records = []
for _, r in chal_df.iterrows():
    p = frozen_predictor.predict_question(r)
    chal_records.append({
        "题号": p["id"], "预测选项": p["predicted_option"],
        "预测词": p["predicted_word"], "是否并列": p["is_tie"],
        "并列选项": p["tied_options"], "分析重点": r["analysis_focus"]
    })
pd.DataFrame(chal_records).to_csv(results_dir / "challenge_analysis.csv", index=False, encoding="utf-8-sig")

self_questions_data = [
    {"id": "S01", "sentence": "工作人员 定期 ____ 工作 安排", "option_A": "发布", "option_B": "提交", "option_C": "召开", "option_D": "参加", "expected": "A", "reason": "匹配训练集'部门 定期 发布 工作 安排'"},
    {"id": "S02", "sentence": "工作人员 定期 ____ 例会", "option_A": "发布", "option_B": "提交", "option_C": "召开", "option_D": "参加", "expected": "C", "reason": "左侧与S01同，右侧匹配'部门 定期 召开 例会'"},
    {"id": "S03", "sentence": "团队 准备 ____ 新 的 设备", "option_A": "安装", "option_B": "排查", "option_C": "遵守", "option_D": "保护", "expected": "A", "reason": "匹配'团队 准备 安装 新 的 设备'"},
    {"id": "S04", "sentence": "团队 需要 ____ 项目 计划", "option_A": "提交", "option_B": "召开", "option_C": "安装", "option_D": "遵守", "expected": "A", "reason": "匹配'团队 需要 提交 项目 计划'"},
    {"id": "S05", "sentence": "教师 应当 ____ 学校 规定", "option_A": "遵守", "option_B": "保护", "option_C": "发布", "option_D": "提交", "expected": "A", "reason": "多合理场景，语料有'教师 带头 遵守 学校 规定'"},
    {"id": "S06", "sentence": "研讨会 将于 明天 ____", "option_A": "召开", "option_B": "参加", "option_C": "发布", "option_D": "提交", "expected": "A", "reason": "句末空位考察 </s> 预测，匹配'会议 将于 明天 召开'"},
]
self_eval_rows = []
for item in self_questions_data:
    p = frozen_predictor.predict_question(item)
    self_eval_rows.append({
        "编号": item["id"], "缺词句": item["sentence"],
        "预先可接受选项": item["expected"], "模型预测": p["predicted_option"],
        "预测词语": p["predicted_word"], "是否并列": p["is_tie"],
        "与预先判断一致": "一致" if p["predicted_option"] == item["expected"] else "不一致"
    })
pd.DataFrame(self_eval_rows).to_csv(results_dir / "self_questions.csv", index=False, encoding="utf-8-sig")

c04_p = frozen_predictor.predict_question(chal_df[chal_df["id"] == "C04"].iloc[0])
print("\n子任务6 - 挑战题与自编题评测:")
print(f"  挑战题 C04: 四候选全 OOV, 预测并列选项: {c04_p['tied_options']}")
s_acc = sum(1 for r in self_eval_rows if r["与预先判断一致"] == "一致")
print(f"  自编题 S01-S06: 预测准确率 {s_acc}/{len(self_eval_rows)} ({s_acc/len(self_eval_rows)*100:.1f}%)")
print(f"  评测结果已保存至 {results_dir / 'challenge_analysis.csv'} 与 {results_dir / 'self_questions.csv'}")
