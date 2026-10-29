# Natural-Language-Processing (自然语言处理课程实验库)

本仓库包含自然语言处理（NLP）课程实验与项目实战的完整代码、实验数据集、输出结果与实验报告。所有代码均已通过本地环境与测试集严格验证。

---

## 实验导航与模块架构

```text
.
├── Test.1/                  # 实验1：NLP 前置技术与开发环境搭建
│   ├── Exp1.ipynb           # 环境与基础分词验证 Notebook
│   ├── check_env.py         # 自动化环境与依赖检测脚本
│   └── Result/              # 环境检测与分词运行证据
│
├── Test.2/                  # 实验2：词向量学习与分布式语义分析
│   ├── task.1/ ~ task.6/    # 独热表示、共现SVD、Skip-gram、CBOW、PCA等独立子任务
│   ├── Result_p/            # 损失收敛曲线、散点图与终端输出证据
│   └── NLP实验2_词向量_...    # 实验 Notebook 与对应报告
│
└── Test.3/                  # 实验3：统计语言模型与中文句子缺词补全
    └── 实验3_统计语言模型/
        ├── NLP实验3_统计语言模型_2429020009_荆文泽.ipynb  # 实验主 Notebook（12单元格，含完整运行输出）
        ├── run_all.py       # 综合汇总主执行程序（分6大任务板块一键运行）
        ├── data/            # 实验原始语料（train.txt / dev.csv / test.csv / challenge.csv）
        ├── results/         # 实验评测产物（模型对比、测试集预测明细、拓展评测结果）
        ├── task1/           # 任务一：统计语言模型的理论到实现（N-gram计数/条件概率/平滑归一化）
        │   ├── task1.py
        │   └── task1_model_comparison.csv
        └── task2/           # 任务二：缺词补全端到端工程实践（面向对象架构）
            ├── lm_core.py   # N-gram 抽象类与 ClozePredictor 局部双向推断引擎
            └── task2.py     # 验证集调优、测试集评测（100%命中）与拓展场景评测
```

---

## 运行环境

- **操作系统**：Windows 11 (64位) / Linux
- **Python 版本**：Python 3.10+
- **核心依赖**：
  ```bash
  pip install numpy pandas scipy
  ```

---

## 实验三执行指南

在 `Test.3/实验3_统计语言模型/` 目录下：

1. **一键运行全部 6 大任务板块（综合汇总）**：
   ```bash
   cd Test.3/实验3_统计语言模型
   python run_all.py
   ```
2. **分模块独立运行**：
   - 任务一（理论验证与平滑归一化校验）：
     ```bash
     cd task1
     python task1.py
     ```
   - 任务二（缺词补全系统与端到端评测）：
     ```bash
     cd task2
     python task2.py
     ```
3. **查看评测产物与 Notebook**：
   - 实验主 Notebook：[NLP实验3_统计语言模型_2429020009_荆文泽.ipynb](file:///D:/NLP/Test.3/实验3_统计语言模型/NLP实验3_统计语言模型_2429020009_荆文泽.ipynb)（包含 12 个单元格，交替呈现 6 大板块任务说明与完整输出结果）。
   - 运行后模型对比、测试集盲评（100% 准确率）、挑战题 OOV 诊断与自编题结果将自动保存至 `results/` 目录。
