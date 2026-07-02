"""
DKT-PyTorch 主程序

Deep Knowledge Tracing (深度知识追踪) 的 PyTorch 实现

使用方法:
    # 基本训练
    python main.py

    # 指定参数训练
    python main.py --epochs 100 --batch_size 64 --lr 0.01

    # 使用普通 RNN (不使用 LSTM)
    python main.py --rnn

    # 从检查点恢复训练
    python main.py --resume output/models/checkpoint_epoch_10.pt
"""

import os
import sys

# 确保从项目根目录运行时可以导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import main

if __name__ == "__main__":
    main()
