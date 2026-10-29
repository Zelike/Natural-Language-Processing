import math
from collections import Counter
from pathlib import Path
import pandas as pd

class NgramLanguageModel:
    def __init__(self, train_sentences):
        self.train_sentences = train_sentences
        self.ordinary_vocab = sorted(list({w for sent in train_sentences for w in sent}))
        self.ordinary_vocab_set = set(self.ordinary_vocab)
        # 统一预测词表：普通词 + </s> + <unk>，不含 <s> 和 ____
        self.pred_vocab = sorted(self.ordinary_vocab) + ["</s>", "<unk>"]
        self.V_size = len(self.pred_vocab)
        
        # 统计频次
        self._count_ngrams()

    def _count_ngrams(self):
        # 1. Unigram
        self.uni_counts = Counter()
        for sent in self.train_sentences:
            for w in sent:
                self.uni_counts[w] += 1
            self.uni_counts["</s>"] += 1
        self.total_unigrams = sum(self.uni_counts.values())

        # 2. Bigram
        self.bi_counts = Counter()
        self.bi_hist = Counter()
        for sent in self.train_sentences:
            seq = ["<s>"] + sent + ["</s>"]
            for h, w in zip(seq[:-1], seq[1:]):
                self.bi_counts[(h, w)] += 1
                self.bi_hist[h] += 1

        # 3. Trigram
        self.tri_counts = Counter()
        self.tri_hist = Counter()
        for sent in self.train_sentences:
            seq = ["<s>", "<s>"] + sent + ["</s>"]
            for h1, h2, w in zip(seq[:-2], seq[1:-1], seq[2:]):
                self.tri_counts[((h1, h2), w)] += 1
                self.tri_hist[(h1, h2)] += 1

    def map_oov(self, word):
        """将词表外普通词统一映射为 <unk>"""
        return word if word in self.ordinary_vocab_set else "<unk>"

    def cond_prob_smoothed(self, history, word, order=2, k=1.0):
        """正 K 平滑条件概率计算"""
        w_mapped = self.map_oov(word) if word != "</s>" else "</s>"
        if order == 1:
            c_hw = self.uni_counts.get(w_mapped, 0)
            c_h = self.total_unigrams
            return (c_hw + k) / (c_h + k * self.V_size)
        elif order == 2:
            h_mapped = history if history == "<s>" else self.map_oov(history)
            c_hw = self.bi_counts.get((h_mapped, w_mapped), 0)
            c_h = self.bi_hist.get(h_mapped, 0)
            return (c_hw + k) / (c_h + k * self.V_size)
        elif order == 3:
            h1 = history[0] if history[0] == "<s>" else self.map_oov(history[0])
            h2 = history[1] if history[1] == "<s>" else self.map_oov(history[1])
            c_hw = self.tri_counts.get(((h1, h2), w_mapped), 0)
            c_h = self.tri_hist.get((h1, h2), 0)
            return (c_hw + k) / (c_h + k * self.V_size)

    def cond_prob_interpolated(self, h_tri, h_bi, word, k=0.5, weights=(0.1, 0.4, 0.5)):
        """三元对齐位置概率线性插值"""
        l1, l2, l3 = weights
        p_u = self.cond_prob_smoothed((), word, order=1, k=k)
        p_b = self.cond_prob_smoothed(h_bi, word, order=2, k=k)
        p_t = self.cond_prob_smoothed(h_tri, word, order=3, k=k)
        return l1 * p_u + l2 * p_b + l3 * p_t

    def score_sentence(self, words, model_type="bigram", k=1.0, weights=(0.1, 0.4, 0.5)):
        """
        计算句子的自然对数似然得分 log P(S) = sum log P(w_i | h_i)
        包含句末 </s> 预测，不包含 <s> 预测
        """
        mapped_words = [self.map_oov(w) for w in words]
        target_words = mapped_words + ["</s>"]
        log_score = 0.0

        if model_type == "unigram":
            for w in target_words:
                p = self.cond_prob_smoothed((), w, order=1, k=k)
                log_score += math.log(p)

        elif model_type == "bigram":
            seq = ["<s>"] + mapped_words
            for h, w in zip(seq, target_words):
                p = self.cond_prob_smoothed(h, w, order=2, k=k)
                log_score += math.log(p)

        elif model_type == "trigram":
            seq_h1 = ["<s>", "<s>"] + mapped_words[:-1] if len(mapped_words) > 0 else ["<s>", "<s>"]
            seq_h2 = ["<s>"] + mapped_words
            for idx in range(len(target_words)):
                h = (seq_h1[idx], seq_h2[idx])
                w = target_words[idx]
                p = self.cond_prob_smoothed(h, w, order=3, k=k)
                log_score += math.log(p)

        elif model_type == "interpolation":
            seq_bi = ["<s>"] + mapped_words
            seq_tri_1 = ["<s>", "<s>"] + (mapped_words[:-1] if len(mapped_words) > 0 else [])
            seq_tri_2 = ["<s>"] + mapped_words
            for idx in range(len(target_words)):
                w = target_words[idx]
                h_b = seq_bi[idx]
                h_t = (seq_tri_1[idx], seq_tri_2[idx])
                p = self.cond_prob_interpolated(h_t, h_b, w, k=k, weights=weights)
                log_score += math.log(p)

        return log_score


class ClozePredictor:
    def __init__(self, lm: NgramLanguageModel, model_type="bigram", k=1.0, weights=(0.1, 0.4, 0.5), tol=1e-10):
        self.lm = lm
        self.model_type = model_type
        self.k = k
        self.weights = weights
        self.tol = tol

    def predict_question(self, question_row):
        """
        对单道缺词补全题目进行四候选替换评分与排序判定
        """
        raw_sentence = question_row["sentence"]
        tokens = raw_sentence.split()
        
        scores = {}
        completed_sentences = {}
        for opt in ["A", "B", "C", "D"]:
            cand = str(question_row[f"option_{opt}"]).strip()
            # 填入候选词 (保留原词展示)
            cand_tokens = [cand if t == "____" else t for t in tokens]
            completed_sentences[opt] = " ".join(cand_tokens)
            scores[opt] = self.lm.score_sentence(
                cand_tokens,
                model_type=self.model_type,
                k=self.k,
                weights=self.weights
            )

        # 排序与并列判定
        sorted_opts = sorted(["A", "B", "C", "D"], key=lambda opt: scores[opt], reverse=True)
        max_score = scores[sorted_opts[0]]
        
        # 识别与最高分并列的所有选项 (在浮点容差 tol 内)
        tied_top_opts = [opt for opt in ["A", "B", "C", "D"] if abs(scores[opt] - max_score) <= self.tol]
        is_tie = len(tied_top_opts) > 1
        
        # 确定性选择：若并列，按 A、B、C、D 顺序取第一个
        chosen_opt = tied_top_opts[0]
        chosen_word = str(question_row[f"option_{chosen_opt}"]).strip()
        
        ranking_str = " > ".join(f"{opt}({scores[opt]:.4f})" for opt in sorted_opts)

        return {
            "id": question_row["id"],
            "predicted_option": chosen_opt,
            "predicted_word": chosen_word,
            "completed_sentence": completed_sentences[chosen_opt],
            "score_A": scores["A"],
            "score_B": scores["B"],
            "score_C": scores["C"],
            "score_D": scores["D"],
            "ranking": ranking_str,
            "is_tie": is_tie,
            "tied_options": "/".join(tied_top_opts) if is_tie else "",
        }
