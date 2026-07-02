#!/usr/bin/env python3
"""
DKT 训练脚本 - 改进版

参考论文设置:
- 批次大小: 100
- 隐藏层: 200
- Dropout: 0.5
- 学习率: 0.1
- 梯度裁剪: 5.0
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Paths
from data_loader import get_data_loaders
from model_improved import DKTModelImproved
from utils import calculate_auc, calculate_accuracy, save_model, EarlyStopping

def train():
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)

    # 设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    print("=" * 60)

    # 超参数 (参考论文)
    BATCH_SIZE = 100  # 论文使用 100
    LEARNING_RATE = 0.1
    N_HIDDEN = 200
    EPOCHS = 50
    DROPOUT = 0.5
    MAX_GRAD_NORM = 5.0
    N_QUESTIONS = 110

    # 创建输出目录
    os.makedirs(Paths.MODEL_DIR, exist_ok=True)

    # 加载数据
    print("\n[1/5] 加载数据...")
    train_loader, test_loader = get_data_loaders(
        train_path=Paths.TRAIN_DATA,
        test_path=Paths.TEST_DATA,
        n_questions=N_QUESTIONS,
        batch_size=BATCH_SIZE  # 使用更大的批次
    )
    print(f"训练集: {len(train_loader)} 批次, 测试集: {len(test_loader)} 批次")
    print(f"批次大小: {BATCH_SIZE}")

    # 创建模型
    print("\n[2/5] 创建模型 (改进版)...")
    model = DKTModelImproved(
        n_questions=N_QUESTIONS,
        n_hidden=N_HIDDEN,
        n_layers=1,
        dropout=DROPOUT,
        use_lstm=True
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # 损失函数和优化器
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    early_stopping = EarlyStopping(patience=15, mode="max")

    # 训练
    print("\n[3/5] 开始训练 (参考论文设置)...")
    print("=" * 60)
    print(f"批次大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"隐藏层: {N_HIDDEN}")
    print("=" * 60)

    best_auc = 0.0

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()

        total_loss = 0.0
        num_batches = 0
        all_preds, all_targets, all_masks = [], [], []

        for batch in train_loader:
            question_ids = batch["question_ids"].to(device)
            correct = batch["correct"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()

            predictions, targets, mask_out = model.predict(question_ids, correct, mask)

            # 展平
            predictions = predictions.flatten()
            targets = targets.flatten()
            mask_out = mask_out.flatten()

            # 计算损失
            loss = (criterion(predictions, targets) * mask_out).sum() / mask_out.sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            all_preds.append(predictions.detach().cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_masks.append(mask_out.cpu().numpy())

        scheduler.step()

        # 计算指标
        train_auc = calculate_auc(
            np.concatenate(all_preds),
            np.concatenate(all_targets),
            np.concatenate(all_masks)
        )
        train_loss = total_loss / num_batches

        # 验证
        model.eval()
        val_preds, val_targets, val_masks = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                question_ids = batch["question_ids"].to(device)
                correct = batch["correct"].to(device)
                mask = batch["mask"].to(device)
                predictions, targets, mask_out = model.predict(question_ids, correct, mask)
                val_preds.append(predictions.flatten().cpu().numpy())
                val_targets.append(targets.flatten().cpu().numpy())
                val_masks.append(mask_out.flatten().cpu().numpy())

        val_auc = calculate_auc(
            np.concatenate(val_preds),
            np.concatenate(val_targets),
            np.concatenate(val_masks)
        )
        val_acc = calculate_accuracy(
            np.concatenate(val_preds),
            np.concatenate(val_targets),
            np.concatenate(val_masks)
        )

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train AUC: {train_auc:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            save_model(model, optimizer, epoch, {"auc": val_auc, "accuracy": val_acc},
                      Paths.MODEL_DIR, "best_model_improved.pt")
            print(f"         >>> 保存最佳模型 (AUC: {best_auc:.4f})")

        if early_stopping(val_auc):
            print(f"\n早停触发!")
            break

    # 完成
    print("\n" + "=" * 60)
    print(f"[4/5] 训练完成! 最佳验证 AUC: {best_auc:.4f}")

    # 测试
    print("\n[5/5] 测试集评估...")
    checkpoint = torch.load(f"{Paths.MODEL_DIR}best_model_improved.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds, test_targets, test_masks = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            question_ids = batch["question_ids"].to(device)
            correct = batch["correct"].to(device)
            mask = batch["mask"].to(device)
            predictions, targets, mask_out = model.predict(question_ids, correct, mask)
            test_preds.append(predictions.flatten().cpu().numpy())
            test_targets.append(targets.flatten().cpu().numpy())
            test_masks.append(mask_out.flatten().cpu().numpy())

    test_auc = calculate_auc(
        np.concatenate(test_preds),
        np.concatenate(test_targets),
        np.concatenate(test_masks)
    )
    test_acc = calculate_accuracy(
        np.concatenate(test_preds),
        np.concatenate(test_targets),
        np.concatenate(test_masks)
    )

    print(f"\n测试集结果:")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print("=" * 60)
    print("\n训练完成!")

if __name__ == "__main__":
    train()
