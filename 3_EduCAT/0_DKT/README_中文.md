# Deep Knowledge Tracing (深度知识追踪)

本项目是论文 **Deep Knowledge Tracing** 的源代码。
论文链接: http://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf

**注意**: 当前代码仅包含 RNN 模型，未包含 LSTM 模型（后续会补充）。

---

## 常见问题解答

**Q: 如何处理训练期间多个学生的数据？看起来不同学生的序列（不同长度的数据）被填充到了相同长度。**

A: 是的，这是为了训练速度而做的处理。

**Q: 训练代码有终止条件吗？**

A: 没有。每一轮训练后都会保存一份模型副本，训练会一直运行直到你手动终止。你可以随时从任意保存的模型继续训练。

---

## 项目结构

```
DeepKnowledgeTracing-master/
├── README.md                    # 本文档
├── scripts/                     # 核心代码
│   ├── rnn.lua                  # RNN 模型定义
│   ├── util.lua                 # 工具函数
│   ├── utilExp.lua              # 实验工具
│   ├── dataSynthetic.lua        # 合成数据加载器
│   ├── dataAssist.lua           # ASSISTments 数据加载器
│   ├── trainSynthetic.lua       # 合成数据训练脚本
│   └── trainAssist.lua          # ASSISTments 数据训练脚本
├── data/                        # 数据集
│   ├── synthetic/               # 合成数据集 (c2/c5: 概念数, q50: 50题, s4000: 4000学生)
│   │   ├── naive_c2_q50_s4000_v*.csv    # 2个概念的合成数据
│   │   ├── naive_c5_q50_s4000_v*.csv    # 5个概念的合成数据
│   │   └── info/                        # 数据集信息
│   └── assistments/             # ASSISTments 真实数据集
│       ├── builder_train.csv    # 训练集
│       └── builder_test.csv     # 测试集
└── output/                      # 输出目录 (训练后生成)
    └── trainRNNAssist/          # 模型保存位置
```

---

## 环境配置 (复现指南)

### 1. 安装 Torch7

这是一个基于 **Lua + Torch7** 的深度学习项目。Torch7 已停止维护，但仍有 Docker 镜像可用。

**方法一：使用 Docker (推荐)**

```bash
# 拉取 Torch7 镜像
docker pull torch/torch7:latest

# 运行容器
docker run -it -v $(pwd):/workspace torch/torch7:latest bash
```

**方法二：Ubuntu/Linux 本地安装**

```bash
# 安装依赖
sudo apt-get update
sudo apt-get install -y git cmake curl wget libreadline-dev libgtk2.0-dev \
    libncurses5-dev pkg-config libqt4-dev libcairo2-dev python-pip python-numpy

# 克隆并安装 Torch7
cd ~
git clone https://github.com/torch/distro.git ~/torch
cd ~/torch
./install-deps
./install.sh

# 激活环境
source ~/.bashrc
torch
```

**方法三：macOS 安装**

```bash
# 使用 brew 安装
brew tap torch/torch
brew install torch7
```

### 2. 安装必要的 Lua 依赖

进入 Torch7 环境后，安装所需的包：

```bash
# 打开 torch REPL
th

# 安装依赖包
luarocks install nn
luarocks install nngraph
luarocks install optim
luarocks install lfs  # LuaFileSystem
luarocks install class # Lua class 库
```

### 3. 数据说明

项目中已包含两个数据集：

1. **合成数据 (Synthetic)**
   - `naive_cX_q50_s4000_vY.csv`: X个概念, 50道题, 4000名学生, 版本Y
   - 可通过修改 `trainSynthetic.lua` 中的 `CONCEPT_NUM` 和 `VERSION` 选择

2. **ASSISTments 数据**
   - 真实教育数据，包含学生答题记录
   - `builder_train.csv` 和 `builder_test.csv`

### 4. 运行训练

**训练合成数据:**

```bash
cd scripts
th trainSynthetic.lua
```

**训练 ASSISTments 数据:**

```bash
cd scripts
th trainAssist.lua
```

模型会保存在 `output/` 目录下。

---

## 关键超参数说明

| 参数 | 说明 | 合成数据默认值 | ASSISTments 默认值 |
|------|------|---------------|-------------------|
| `n_hidden` | 隐藏层神经元数 | 400 | 200 |
| `mini_batch_size` | Mini-batch 大小 | 50 | 100 |
| `init_rate` | 初始学习率 | 0.5 | 30 |
| `dropout` | 是否使用 Dropout | true | true |
| `max_grad` | 梯度裁剪阈值 | 100 | 5e-5 |

---

## 注意事项

1. **macOS Apple Silicon (M1/M2/M3)**: Torch7 不支持 Apple Silicon，建议使用 Docker 或考虑使用 Python 版本的 DKT 实现（如 PyTorch）。

2. **Python 替代方案**: 如果遇到环境问题，可以使用 Python 重新实现 DKT，核心架构为：
   - 输入层 → LSTM/GRU → 全连接层 → Sigmoid 输出

3. **训练时间**: 合成数据训练速度较快；ASSISTments 数据可能需要数小时。

---

## 参考

- 原论文: Piech et al., "Deep Knowledge Tracing", NeurIPS 2015
- 论文链接: http://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf
