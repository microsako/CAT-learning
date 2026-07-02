"""
DKT 模型训练脚本

Deep Knowledge Tracing 的完整训练流程，包括:
- 数据加载
- 模型构建
- 训练循环
- 验证与测试
- 模型保存

使用方法:
    python train.py

可选参数:
    --epochs: 训练轮数
    --batch_size: 批次大小
    --lr: 学习率
    --hidden: 隐藏层大小
    --device: 计算设备 (cuda/mps/cpu)
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 导入项目模块
from config import Paths, ModelConfig, TrainConfig, DataConfig, DeviceConfig
from data_loader import get_data_loaders
from model import DKTModel
from utils import (
    evaluate_model,
    save_model,
    EarlyStopping,
    print_metrics,
    count_parameters,
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DKT 训练脚本")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=TrainConfig.EPOCHS,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=TrainConfig.BATCH_SIZE,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=TrainConfig.LEARNING_RATE,
                        help="学习率")
    parser.add_argument("--hidden", type=int, default=ModelConfig.N_HIDDEN,
                        help="隐藏层大小")

    # 模型参数
    parser.add_argument("--lstm", action="store_true", default=ModelConfig.USE_LSTM,
                        help="使用 LSTM (默认)")
    parser.add_argument("--rnn", action="store_true",
                        help="使用普通 RNN")
    parser.add_argument("--dropout", type=float, default=ModelConfig.DROPOUT,
                        help="Dropout 概率")

    # 其他参数
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备 (cuda/mps/cpu)")
    parser.add_argument("--seed", type=int, default=DeviceConfig.RANDOM_SEED,
                        help="随机种子")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1,
                        help="模型保存间隔")

    # 从检查点恢复
    parser.add_argument("--resume", type=str, default=None,
                        help="从检查点恢复训练")

    return parser.parse_args()


def set_random_seed(seed: int):
    """
    设置随机种子保证可复现性

    Args:
        seed: 随机种子值
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(device_str: str = None) -> torch.device:
    """
    获取计算设备

    优先级: MPS > CUDA > CPU

    Args:
        device_str: 手动指定设备

    Returns:
        torch.device 对象
    """
    if device_str:
        return torch.device(device_str)

    # 检查 MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        print("检测到 Apple Silicon GPU，使用 MPS 加速")
        return torch.device("mps")

    # 检查 CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        print(f"检测到 NVIDIA GPU，使用 CUDA 加速")
        return torch.device("cuda")

    # 使用 CPU
    print("使用 CPU 计算")
    return torch.device("cpu")


def train_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 5.0,
    log_interval: int = 10
) -> dict:
    """
    训练一个 epoch

    Args:
        model: DKT 模型
        data_loader: 数据加载器
        criterion: 损失函数 (BCELoss)
        optimizer: 优化器
        device: 计算设备
        max_grad_norm: 梯度裁剪阈值
        log_interval: 日志打印间隔

    Returns:
        包含训练指标的字典
    """
    model.train()  # 设置为训练模式

    total_loss = 0.0
    num_batches = 0
    total_samples = 0

    # 用于计算 AUC 的累积
    all_predictions = []
    all_targets = []
    all_masks = []

    epoch_start = time.time()

    for batch_idx, batch in enumerate(data_loader):
        # 移动到设备
        question_ids = batch["question_ids"].to(device)
        correct = batch["correct"].to(device)
        mask = batch["mask"].to(device)

        # 前向传播
        optimizer.zero_grad()

        # 获取预测
        # predictions: (batch, seq_len - 1)
        predictions, targets, mask_out = model.predict(question_ids, correct, mask)

        # 计算损失 (只考虑有效位置)
        # 使用二元交叉熵损失
        loss = criterion(predictions, targets)
        masked_loss = (loss * mask_out).sum() / mask_out.sum()

        # 反向传播
        masked_loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # 更新参数
        optimizer.step()

        # 记录损失
        total_loss += masked_loss.item()
        num_batches += 1
        total_samples += mask_out.sum().item()

        # 收集预测结果用于计算 AUC
        all_predictions.append(predictions.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        all_masks.append(mask_out.detach().cpu().numpy())

        # 打印日志
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            print(f"  Batch {batch_idx + 1}/{len(data_loader)}: "
                  f"Loss={avg_loss:.6f}")

    epoch_time = time.time() - epoch_start

    # 计算整体指标
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    all_masks = np.concatenate(all_masks, axis=0).flatten()

    # 计算 AUC
    from utils import calculate_auc, calculate_accuracy
    auc = calculate_auc(all_predictions, all_targets, all_masks)
    accuracy = calculate_accuracy(all_predictions, all_targets, all_masks)

    metrics = {
        "loss": total_loss / num_batches,
        "auc": auc,
        "accuracy": accuracy,
        "num_samples": total_samples,
        "time": epoch_time
    }

    return metrics


def validate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """
    在验证集上评估模型

    Args:
        model: DKT 模型
        data_loader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备

    Returns:
        包含验证指标的字典
    """
    model.eval()  # 设置为评估模式

    total_loss = 0.0
    num_batches = 0

    all_predictions = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in data_loader:
            question_ids = batch["question_ids"].to(device)
            correct = batch["correct"].to(device)
            mask = batch["mask"].to(device)

            # 预测
            predictions, targets, mask_out = model.predict(question_ids, correct, mask)

            # 计算损失
            loss = criterion(predictions, targets)
            masked_loss = (loss * mask_out).sum() / mask_out.sum()

            total_loss += masked_loss.item()
            num_batches += 1

            # 收集结果
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_masks.append(mask_out.cpu().numpy())

    # 计算指标
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    all_masks = np.concatenate(all_masks, axis=0).flatten()

    from utils import calculate_auc, calculate_accuracy
    auc = calculate_auc(all_predictions, all_targets, all_masks)
    accuracy = calculate_accuracy(all_predictions, all_targets, all_masks)

    metrics = {
        "loss": total_loss / num_batches,
        "auc": auc,
        "accuracy": accuracy
    }

    model.train()  # 恢复训练模式

    return metrics


def main():
    """主训练函数"""
    # 解析参数
    args = parse_args()

    # 设置随机种子
    set_random_seed(args.seed)

    # 获取设备
    device = get_device(args.device)
    print(f"\n使用设备: {device}")

    # 创建输出目录
    os.makedirs(Paths.MODEL_DIR, exist_ok=True)
    os.makedirs(Paths.LOG_DIR, exist_ok=True)

    # 加载数据
    print("\n" + "=" * 60)
    print("加载数据...")
    print("=" * 60)

    train_loader, test_loader = get_data_loaders(
        train_path=Paths.TRAIN_DATA,
        test_path=Paths.TEST_DATA,
        n_questions=DataConfig.N_QUESTIONS,
        max_seq_len=DataConfig.MAX_SEQ_LEN,
        batch_size=args.batch_size
    )

    # 创建模型
    print("\n" + "=" * 60)
    print("创建模型...")
    print("=" * 60)

    use_rnn = args.rnn
    model = DKTModel(
        n_questions=DataConfig.N_QUESTIONS,
        n_hidden=args.hidden,
        n_layers=1,
        dropout=args.dropout,
        use_lstm=not use_rnn,
        use_compressed_sensing=DataConfig.USE_COMPRESSED_SENSING,
        compressed_dim=DataConfig.COMPRESSED_DIM
    )

    model = model.to(device)

    num_params = count_parameters(model)
    print(f"模型参数数量: {num_params:,}")
    print(f"使用 {'LSTM' if not use_rnn else 'RNN'}")

    # 定义损失函数和优化器
    criterion = nn.BCELoss(reduction='none')  # 不进行 reduction，手动应用掩码
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 学习率调度器 (可选)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=TrainConfig.LEARNING_RATE_DECAY
    )

    # 早停
    early_stopping = EarlyStopping(
        patience=TrainConfig.EARLY_STOPPING_PATIENCE,
        mode="max"
    )

    # TensorBoard 日志
    writer = SummaryWriter(Paths.LOG_DIR)

    # 恢复训练
    start_epoch = 0
    if args.resume:
        print(f"\n从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"从 Epoch {start_epoch} 继续训练")

    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    best_val_auc = 0.0
    best_model_path = None

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # 训练
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            max_grad_norm=TrainConfig.MAX_GRAD_NORM,
            log_interval=args.log_interval
        )

        # 验证
        val_metrics = validate(model, test_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # 打印结果
        print(f"\n训练结果:")
        print(f"  Loss: {train_metrics['loss']:.6f}")
        print(f"  AUC:  {train_metrics['auc']:.4f}")
        print(f"  Acc:  {train_metrics['accuracy']:.4f}")
        print(f"  用时: {epoch_time:.2f}s")

        print(f"\n验证结果:")
        print(f"  Loss: {val_metrics['loss']:.6f}")
        print(f"  AUC:  {val_metrics['auc']:.4f}")
        print(f"  Acc:  {val_metrics['accuracy']:.4f}")

        # 记录到 TensorBoard
        writer.add_scalar("Train/Loss", train_metrics["loss"], epoch)
        writer.add_scalar("Train/AUC", train_metrics["auc"], epoch)
        writer.add_scalar("Train/Accuracy", train_metrics["accuracy"], epoch)
        writer.add_scalar("Val/Loss", val_metrics["loss"], epoch)
        writer.add_scalar("Val/AUC", val_metrics["auc"], epoch)
        writer.add_scalar("Val/Accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

        # 保存最佳模型
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_model_path = save_model(
                model, optimizer, epoch, val_metrics,
                Paths.MODEL_DIR, "best_model.pt"
            )
            print(f"\n>>> 保存最佳模型 (AUC: {best_val_auc:.4f})")

        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            save_model(
                model, optimizer, epoch, val_metrics,
                Paths.MODEL_DIR, f"checkpoint_epoch_{epoch + 1}.pt"
            )

        # 学习率调整
        scheduler.step()

        # 早停检查
        if early_stopping(val_metrics["auc"]):
            print(f"\n早停触发! 连续 {early_stopping.patience} 轮验证 AUC 没有改善")
            break

    # 训练结束
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"最佳验证 AUC: {best_val_auc:.4f}")
    print(f"最佳模型保存于: {best_model_path}")

    # 在测试集上评估最佳模型
    print("\n" + "=" * 60)
    print("在测试集上评估最佳模型...")
    print("=" * 60)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = validate(model, test_loader, criterion, device)
    print(f"\n测试结果:")
    print(f"  Loss: {test_metrics['loss']:.6f}")
    print(f"  AUC:  {test_metrics['auc']:.4f}")
    print(f"  Acc:  {test_metrics['accuracy']:.4f}")

    # 关闭 TensorBoard writer
    writer.close()

    print("\n" + "=" * 60)
    print("使用方法:")
    print("  1. 启动 TensorBoard: tensorboard --logdir=output/logs")
    print("  2. 打开浏览器访问: http://localhost:6006")
    print("  3. 查看训练曲线和评估指标")
    print("=" * 60)


if __name__ == "__main__":
    main()
