"""
工具函数模块

包含评估指标计算、模型保存/加载等辅助函数。
"""

import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, accuracy_score


def calculate_auc(predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    """
    计算 AUC (Area Under the ROC Curve)

    AUC 是 DKT 论文中主要使用的评估指标，衡量模型区分正负样本的能力。
    AUC = 0.5 表示随机猜测，AUC = 1.0 表示完美预测。

    注意: 只有当 mask 中有至少两个不同类别时才计算 AUC

    Args:
        predictions: 预测概率 (flattened)
        targets: 真实标签 (flattened)
        mask: 有效位置掩码 (flattened)

    Returns:
        AUC 值，范围 [0, 1]
    """
    # 应用掩码
    valid_indices = mask > 0
    pred = predictions[valid_indices]
    true = targets[valid_indices]

    # 如果只有一个类别，返回 0.5
    if len(np.unique(true)) < 2:
        return 0.5

    try:
        auc = roc_auc_score(true, pred)
        return auc
    except ValueError:
        # 如果计算失败，返回 0.5
        return 0.5


def calculate_accuracy(predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    """
    计算准确率

    Args:
        predictions: 预测概率 (flattened)
        targets: 真实标签 (flattened)
        mask: 有效位置掩码 (flattened)

    Returns:
        准确率，范围 [0, 1]
    """
    # 应用掩码
    valid_indices = mask > 0
    pred = (predictions[valid_indices] > 0.5).astype(int)
    true = targets[valid_indices].astype(int)

    accuracy = accuracy_score(true, pred)
    return accuracy


def calculate_rmse(predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    """
    计算 RMSE (Root Mean Square Error)

    Args:
        predictions: 预测概率 (flattened)
        targets: 真实标签 (flattened)
        mask: 有效位置掩码 (flattened)

    Returns:
        RMSE 值
    """
    # 应用掩码
    valid_indices = mask > 0
    pred = predictions[valid_indices]
    true = targets[valid_indices]

    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    return_predictions: bool = False
) -> Dict[str, float]:
    """
    在整个数据集上评估模型

    Args:
        model: DKT 模型
        data_loader: 数据加载器
        device: 计算设备
        return_predictions: 是否返回预测结果

    Returns:
        包含评估指标的字典
    """
    model.eval()  # 设置为评估模式

    all_predictions = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in data_loader:
            # 移动到设备
            question_ids = batch["question_ids"].to(device)
            correct = batch["correct"].to(device)
            mask = batch["mask"].to(device)

            # 预测
            predictions, targets, mask_out = model.predict(question_ids, correct, mask)

            # 收集结果
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_masks.append(mask_out.cpu().numpy())

    # 合并所有批次
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    # 计算指标
    metrics = {
        "auc": calculate_auc(all_predictions, all_targets, all_masks),
        "accuracy": calculate_accuracy(all_predictions, all_targets, all_masks),
        "rmse": calculate_rmse(all_predictions, all_targets, all_masks),
    }

    if return_predictions:
        metrics["predictions"] = all_predictions
        metrics["targets"] = all_targets

    model.train()  # 恢复训练模式

    return metrics


def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_dir: str,
    filename: str = "model.pt"
) -> str:
    """
    保存模型检查点

    Args:
        model: DKT 模型
        optimizer: 优化器
        epoch: 当前轮次
        metrics: 评估指标
        save_dir: 保存目录
        filename: 文件名

    Returns:
        保存的文件路径
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    # 将 metrics 中的 numpy 值转换为 Python 原生类型
    metrics_native = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics_native,
    }

    torch.save(checkpoint, filepath)
    return filepath


def load_model(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    device: torch.device
) -> Tuple[int, Dict[str, float]]:
    """
    加载模型检查点

    Args:
        model: DKT 模型 (权重会被加载到这个模型)
        optimizer: 优化器 (可选，权重会被加载)
        checkpoint_path: 检查点文件路径
        device: 设备

    Returns:
        (epoch, metrics) 元组
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})

    return epoch, metrics


class MetricsTracker:
    """
    训练指标追踪器

    记录每个 epoch 的训练和验证指标，支持早停。
    """

    def __init__(self, patience: int = 10, metric: str = "auc"):
        """
        Args:
            patience: 早停耐心值
            metric: 用于早停的指标名
        """
        self.patience = patience
        self.metric = metric
        self.history = {
            "train_loss": [],
            "train_auc": [],
            "train_acc": [],
            "val_loss": [],
            "val_auc": [],
            "val_acc": [],
        }
        self.best_epoch = 0
        self.best_value = -float("inf") if "auc" in metric else float("inf")
        self.counter = 0

    def update(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        更新指标

        Args:
            epoch: 当前轮次
            metrics: 指标字典

        Returns:
            如果触发早停返回 True，否则返回 False
        """
        # 记录历史
        for key in self.history:
            if key in metrics:
                self.history[key].append(metrics[key])

        # 检查是否需要早停
        current_value = metrics.get(self.metric, 0)

        # 根据指标类型判断是否更好
        if "auc" in self.metric or "acc" in self.metric:
            is_better = current_value > self.best_value
        else:
            is_better = current_value < self.best_value

        if is_better:
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    def get_best(self) -> Tuple[int, float]:
        """获取最佳 epoch 和指标值"""
        return self.best_epoch, self.best_value

    def save(self, filepath: str):
        """保存历史记录到文件"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

    def print_summary(self):
        """打印训练摘要"""
        print("\n" + "=" * 60)
        print("训练摘要")
        print("=" * 60)
        print(f"最佳 {self.metric}: {self.best_value:.4f} (Epoch {self.best_epoch + 1})")
        print(f"总训练轮数: {len(self.history['train_loss'])}")
        print(f"早停轮数: {self.counter}")

        if len(self.history["val_auc"]) > 0:
            print(f"\n验证 AUC 变化:")
            for i, auc in enumerate(self.history["val_auc"]):
                print(f"  Epoch {i+1}: {auc:.4f}")

        print("=" * 60)


class EarlyStopping:
    """
    早停策略

    如果验证集性能连续 N 轮没有改善，则停止训练。
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "max"):
        """
        Args:
            patience: 耐心值
            min_delta: 最小改善量
            mode: "max" 表示越大越好，"min" 表示越小越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        判断是否应该早停

        Args:
            score: 当前分数

        Returns:
            True 表示应该停止训练
        """
        if self.best_score is None:
            self.best_score = score
            return False

        # 判断是否改善
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False


def print_metrics(epoch: int, metrics: Dict[str, float], prefix: str = ""):
    """
    格式化打印指标

    Args:
        epoch: 当前轮次
        metrics: 指标字典
        prefix: 前缀字符串 (如 "Train" 或 "Val")
    """
    if prefix:
        prefix = f"{prefix} "

    loss_str = f"{metrics.get('loss', 0):.6f}" if 'loss' in metrics else "N/A"
    auc_str = f"{metrics.get('auc', 0):.4f}"
    acc_str = f"{metrics.get('accuracy', 0):.4f}"
    rmse_str = f"{metrics.get('rmse', 0):.4f}"

    print(f"Epoch {epoch}: Loss={loss_str}, AUC={auc_str}, Acc={acc_str}, RMSE={rmse_str}")


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型的可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    # 测试指标计算
    print("=" * 50)
    print("测试工具函数")
    print("=" * 50)

    # 创建测试数据
    predictions = np.array([0.1, 0.9, 0.6, 0.3, 0.8, 0.2])
    targets = np.array([0, 1, 1, 0, 1, 0])
    mask = np.array([1, 1, 1, 1, 1, 1])

    # 计算指标
    auc = calculate_auc(predictions, targets, mask)
    acc = calculate_accuracy(predictions, targets, mask)
    rmse = calculate_rmse(predictions, targets, mask)

    print(f"\n测试指标计算:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  RMSE: {rmse:.4f}")

    # 测试早停
    print("\n测试早停:")
    early_stopping = EarlyStopping(patience=3, mode="max")

    for i in range(10):
        score = 0.5 + 0.1 * i if i < 5 else 0.4
        should_stop = early_stopping(score)
        print(f"  Round {i+1}: score={score:.3f}, stop={should_stop}")
        if should_stop:
            break

    print("\n所有测试完成!")
