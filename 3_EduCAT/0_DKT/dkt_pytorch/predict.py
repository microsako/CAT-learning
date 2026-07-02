"""
预测脚本

使用训练好的模型对新的学生数据进行预测。

使用方法:
    python predict.py --model output/models/best_model.pt --input data/student_sequence.csv
"""

import argparse
import numpy as np
import torch
from data_loader import ASSISTmentsDataset
from model import DKTModel
from utils import load_model, calculate_auc, calculate_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="DKT 预测脚本")
    parser.add_argument("--model", type=str, required=True, help="模型文件路径")
    parser.add_argument("--question_ids", type=str, required=True, help="题目ID序列 (逗号分隔)")
    parser.add_argument("--correct", type=str, required=True, help="正确性序列 (逗号分隔)")
    parser.add_argument("--n_questions", type=int, default=110, help="题目总数")
    parser.add_argument("--hidden", type=int, default=200, help="隐藏层大小")
    return parser.parse_args()


def predict_sequence(
    model: DKTModel,
    question_ids: list,
    correct: list,
    device: torch.device
) -> list:
    """
    对单个学生的答题序列进行预测

    Args:
        model: 训练好的 DKT 模型
        question_ids: 题目ID列表
        correct: 正确性列表
        device: 计算设备

    Returns:
        每个位置的预测概率列表
    """
    model.eval()

    # 转换为张量
    q_ids = torch.tensor([question_ids], dtype=torch.long).to(device)
    corr = torch.tensor([correct], dtype=torch.float32).to(device)

    with torch.no_grad():
        result = model(q_ids, corr)
        predictions = result["predictions"][0].cpu().numpy().tolist()

    return predictions


def main():
    args = parse_args()

    # 解析输入
    question_ids = [int(x) + 1 for x in args.question_ids.split(",")]
    correct = [int(x) for x in args.correct.split(",")]

    print(f"输入序列:")
    print(f"  题目ID: {question_ids}")
    print(f"  正确性: {correct}")

    # 加载模型
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    model = DKTModel(
        n_questions=args.n_questions,
        n_hidden=args.hidden,
        use_lstm=True
    ).to(device)

    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"\n模型加载成功: {args.model}")
    print(f"设备: {device}")

    # 预测
    predictions = predict_sequence(model, question_ids, correct, device)

    print(f"\n预测结果 (预测下一题的正确概率):")
    for i, (q_id, corr, pred) in enumerate(zip(question_ids[:-1], correct[1:], predictions)):
        print(f"  位置 {i+1}: 题目 {q_id}, 预测 {pred:.4f}, 实际 {corr}")

    # 计算 AUC (如果有足够的数据)
    if len(correct) > 2:
        # 目标是从第2题开始预测
        targets = correct[1:]
        pred_array = predictions
        mask = np.ones(len(predictions))
        auc = calculate_auc(np.array(predictions), np.array(targets), mask)
        acc = calculate_accuracy(np.array(predictions), np.array(targets), mask)
        print(f"\n整体指标:")
        print(f"  AUC: {auc:.4f}")
        print(f"  Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
