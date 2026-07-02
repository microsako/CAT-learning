# DKT-PyTorch 中文使用指南

## 项目介绍

本项目是 **Deep Knowledge Tracing (深度知识追踪)** 的 PyTorch 实现，原论文来自斯坦福大学 Piech 等人发表于 NeurIPS 2015。

### 什么是 DKT？

DKT 是一种使用深度学习技术追踪学生知识状态的方法。核心思想是：

> 给定学生历史的答题序列，预测学生下一题的正确概率。

例如：如果一个学生之前在代数题目上经常出错，模型会学习到该学生在代数方面的知识掌握较弱，从而预测她在下一道代数题上答对的概率较低。

### 项目结构

```
dkt_pytorch/
├── config.py         # 配置文件 (超参数)
├── data_loader.py    # 数据加载器
├── model.py          # DKT 模型定义
├── utils.py          # 工具函数 (评估指标等)
├── train.py          # 训练脚本
├── main.py           # 程序入口
├── predict.py        # 预测脚本
├── requirements.txt  # 依赖列表
└── README_CN.md      # 本文档
```

---

## 快速开始

### 1. 安装依赖

```bash
# 进入项目目录
cd dkt_pytorch

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行训练

```bash
# 基本训练 (使用默认参数)
python main.py

# 指定参数训练
python main.py --epochs 50 --batch_size 64 --lr 0.01 --hidden 200
```

### 3. 查看结果

训练过程中会打印以下信息：

```
--- Epoch 1/50 ---

训练结果:
  Loss: 0.652341
  AUC:  0.6234
  Acc:  0.6123
  用时: 12.34s

验证结果:
  Loss: 0.648912
  AUC:  0.6312
  Acc:  0.6187
```

#### 指标说明

| 指标 | 说明 | 参考值 |
|------|------|--------|
| **Loss** | 二元交叉熵损失，越小越好 | < 0.55 |
| **AUC** | ROC 曲线下面积，越大越好 (0.5~1.0) | > 0.70 |
| **Acc** | 准确率，正确预测的比例 | > 0.65 |

#### 查看训练曲线

使用 TensorBoard 可视化训练过程：

```bash
# 启动 TensorBoard
tensorboard --logdir=output/logs

# 打开浏览器访问
# http://localhost:6006
```

---

## 参数说明

### 训练参数

```bash
# 训练相关
--epochs      # 训练轮数，默认 50
--batch_size  # 批次大小，默认 32
--lr          # 学习率，默认 0.1

# 模型相关
--hidden      # 隐藏层大小，默认 200
--dropout     # Dropout 概率，默认 0.5
--lstm        # 使用 LSTM (默认)
--rnn         # 使用普通 RNN

# 其他
--device      # 计算设备: cuda/mps/cpu
--seed        # 随机种子，默认 42
```

### 配置修改

编辑 `config.py` 文件可以永久修改默认配置：

```python
class TrainConfig:
    BATCH_SIZE = 32          # 修改批次大小
    LEARNING_RATE = 0.1      # 修改学习率
    EPOCHS = 50              # 修改训练轮数

class ModelConfig:
    N_HIDDEN = 200           # 修改隐藏层大小
    DROPOUT = 0.5            # 修改 Dropout
```

---

## 模型使用

### 加载训练好的模型

```python
import torch
from model import DKTModel
from utils import load_model

# 加载模型
model = DKTModel(n_questions=110, n_hidden=200)
checkpoint = torch.load("output/models/best_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 预测
question_ids = torch.tensor([[1, 5, 8, 12]])  # 题目ID
correct = torch.tensor([[1, 0, 1, 1]])         # 正确性

with torch.no_grad():
    result = model(question_ids, correct)
    predictions = result["predictions"]  # 预测概率
```

### 命令行预测

```bash
python predict.py \
    --model output/models/best_model.pt \
    --question_ids "1,5,8,12" \
    --correct "1,0,1,1"
```

---

## 常见问题

### Q1: 显存不足 (CUDA Out of Memory)

```bash
# 减小批次大小
python main.py --batch_size 16

# 或减小隐藏层大小
python main.py --hidden 100
```

### Q2: Apple Silicon (M1/M2/M3) 报错

确保安装了支持 MPS 的 PyTorch 版本：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/mps
```

### Q3: 训练太慢

- 使用 GPU/MPS 加速
- 增大批次大小
- 减少 `MAX_SEQ_LEN` (在 config.py 中)

### Q4: 模型性能不佳

1. 调整学习率 (试试 0.01, 0.05, 0.1, 0.5)
2. 增加隐藏层大小
3. 增加训练轮数
4. 调整 Dropout (试试 0.3, 0.4, 0.5, 0.6)

---

## 数据格式

### ASSISTments 数据格式

原始数据文件 (`data/assistments/builder_train.csv`) 格式：

```
7
7,7,7,7,7,7,8
1,1,0,1,1,1,1

5
5,5,5,5,5
1,1,1,0,1
```

每 4 行是一个学生：
- 第 1 行：答题数量
- 第 2 行：题目 ID 序列
- 第 3 行：正确性 (0=错误, 1=正确)
- 第 4 行：空行

---

## 参考

- 原论文: [Deep Knowledge Tracing](http://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf)
- PyTorch 官方文档: https://pytorch.org/docs/
