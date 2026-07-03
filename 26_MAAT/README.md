# MAAT — Quality meets Diversity: A Model-Agnostic Framework for CAT

MAAT(ICDM 2020)论文的学习资料 + 官方代码 + 复现实验,一站式文件夹。

> 论文:Bi et al., *Quality meets Diversity: A Model-Agnostic Framework for Computerized Adaptive Testing*, ICDM 2020. arXiv: 2101.05986

## 目录结构

```
26_MAAT/
├── README.md                  # 本文件
├── paper/                     # 论文
│   ├── 2101.05986v1.pdf           # 英文原文
│   └── maat-model-agnostic-cat.zh.pdf  # 整篇中文翻译
├── notes/                     # 笔记与详解(每篇 md 源文件 + pdf 排版稿)
│   ├── MAAT精读笔记               # 全文精读:动机、三模块、实验
│   ├── 多样性模块公式详解          # 式 (5)-(6):IWKC、次模性、贪心
│   ├── 重要性模块公式详解          # 式 (10)-(15):含 Skip-Gram 零基础补课
│   └── MAAT代码全解               # 本仓库代码逐层讲解 + 复现结果
├── datasets/                  # 数据(官方仓库原有)
│   ├── assistment/                # ASSISTments:triplets.csv 作答三元组、
│   │                              #   concept_map.json 题目-知识点映射、metadata.json
│   └── data_prep.py               # 按学生划分训练/测试集的工具
├── models/                    # 预训练权重(官方仓库原有)
│   ├── irt/checkpoint.pt          # IRT 题目参数(alpha/beta),test.py 直接加载
│   └── ncd/checkpoint.pt          # NCD 权重(源码缺失,暂无法使用)
├── pyat/                      # 核心代码包(注释已全部汉化)
│   ├── utils/data/                # 数据集类:_Dataset / TrainDataset / AdapTestDataset
│   ├── model/                     # IRT 模型 + EMC 质量模块 + 抽象接口
│   ├── strategy/                  # 四个选题策略 + 抽象接口
│   └── driver.py                  # 自适应测试主循环
├── scripts/
│   ├── train.py                   # 离线预训练 IRT(已有 checkpoint,可不跑)
│   └── test.py                    # 四策略对比实验入口
├── results/                   # 复现实验输出
│   ├── 2026-07-03-20-58-58/       # 完整四策略逐步指标(results.csv)
│   ├── 策略对比图.png              # AUC / COV 对比图
│   └── run_log.txt                # 运行日志
└── .venv/                     # 实验用虚拟环境(torch 2.12.1 等)
```

## 代码状态说明

官方仓库([bigdata-ustc/MAAT](https://github.com/bigdata-ustc/MAAT))发布不完整,本文件夹做过如下修补(细节见 `notes/MAAT代码全解`第 7 节):

- **补写**三个缺失的策略文件:`random_strategy.py`、`expected_model_change_strategy.py`(官方 `__init__.py` 引用了但文件不存在)、`fisher_strategy.py`(论文基线 MFI,官方完全没有);
- **修复**两处新版 torch/numpy 兼容问题(变长知识点列表无法拼批、β 参数数组转标量);
- **全部注释汉化**;
- 论文的**重要性模块**(式 10–15,知识点权重 $w_k$)官方未开源,代码等价于所有知识点等权;NCD 模型源码同样缺失。

## 快速上手

```bash
source .venv/bin/activate
cd scripts
python test.py        # 四策略对比,CPU 约 20 分钟
                      # 结果写入 results/<时间戳>/results.csv
```

注意:`test.py` 必须在 `scripts/` 目录下运行(数据用相对路径)。

## 复现结果速览

ASSISTments,47 名测试学生,IRT,考 50 题:

| 策略 | AUC@50 | COV@50 |
|---|---|---|
| 随机选题 | 0.6666 | 0.794 |
| Fisher (MFI) | 0.6815 | 0.694 |
| 纯 EMC | 0.6811 | 0.833 |
| **MAAT** | **0.6827** | **0.878** |

论文结论全部复现:MAAT 精度不输传统方法,知识点覆盖率显著最高;Fisher 覆盖率垫底,正是论文批评的"只顾精度不顾覆盖"。

## 推荐阅读顺序

1. `notes/MAAT精读笔记` — 先建立全文框架;
2. `notes/多样性模块公式详解` → `notes/重要性模块公式详解` — 两块数学细节(后者含 word2vec/Skip-Gram 从零补课);
3. `notes/MAAT代码全解` — 对着代码把论文落地,含复现全过程。
