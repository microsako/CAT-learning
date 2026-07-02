#!/usr/bin/env python3
"""Quick test script"""
import sys
import os

# Suppress TensorBoard warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("DKT PyTorch 训练测试")
print("=" * 50)

# Test imports
print("\n1. 测试导入模块...")
try:
    from config import Paths, ModelConfig, TrainConfig, DataConfig, DeviceConfig
    print("   [OK] config.py")
except Exception as e:
    print(f"   [FAIL] config.py: {e}")
    sys.exit(1)

try:
    from data_loader import get_data_loaders
    print("   [OK] data_loader.py")
except Exception as e:
    print(f"   [FAIL] data_loader.py: {e}")
    sys.exit(1)

try:
    from model import DKTModel
    print("   [OK] model.py")
except Exception as e:
    print(f"   [FAIL] model.py: {e}")
    sys.exit(1)

try:
    from utils import calculate_auc, calculate_accuracy
    print("   [OK] utils.py")
except Exception as e:
    print(f"   [FAIL] utils.py: {e}")
    sys.exit(1)

# Test device
print("\n2. 检测计算设备...")
import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"   [OK] 使用 MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"   [OK] 使用 CUDA (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print(f"   [OK] 使用 CPU")

# Test data loading
print("\n3. 加载数据...")
try:
    train_loader, test_loader = get_data_loaders(
        train_path=Paths.TRAIN_DATA,
        test_path=Paths.TEST_DATA,
        n_questions=DataConfig.N_QUESTIONS,
        batch_size=32
    )
    print(f"   [OK] 训练集: {len(train_loader)} 批次")
    print(f"   [OK] 测试集: {len(test_loader)} 批次")
except Exception as e:
    print(f"   [FAIL] 数据加载: {e}")
    sys.exit(1)

# Test model
print("\n4. 创建模型...")
try:
    model = DKTModel(
        n_questions=DataConfig.N_QUESTIONS,
        n_hidden=ModelConfig.N_HIDDEN,
        n_layers=1,
        dropout=ModelConfig.DROPOUT,
        use_lstm=ModelConfig.USE_LSTM
    ).to(device)
    print(f"   [OK] 模型创建成功")
    print(f"   [OK] 参数量: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"   [FAIL] 模型创建: {e}")
    sys.exit(1)

# Quick forward pass test
print("\n5. 测试前向传播...")
try:
    batch = next(iter(train_loader))
    question_ids = batch["question_ids"].to(device)
    correct = batch["correct"].to(device)
    mask = batch["mask"].to(device)

    predictions, targets, mask_out = model.predict(question_ids, correct, mask)
    print(f"   [OK] 输入形状: {question_ids.shape}")
    print(f"   [OK] 输出形状: {predictions.shape}")
    print(f"   [OK] 预测范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
except Exception as e:
    print(f"   [FAIL] 前向传播: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("所有测试通过！准备开始训练...")
print("=" * 50)
print("""
下一步: 运行训练
  python main.py

或使用参数:
  python main.py --epochs 50 --batch_size 32 --lr 0.1
""")
