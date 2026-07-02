"""
EduCAT 快速训练脚本

这个脚本演示了如何使用 EduCAT 库训练一个最简单的 CAT 模型（IRT模型）。

运行方式：
    cd EduCAT
    python run_train.py
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CAT.model import IRTModel
from CAT.strategy import MFIStrategy
from CAT.dataset import TrainDataset, AdapTestDataset

import json
import numpy as np
import random

# 设置随机种子
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def setuplogger():
    """设置日志"""
    import logging
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    return root

def generate_demo_data(num_train_students=80, num_test_students=20, num_questions=50, num_concepts=10):
    """
    生成演示数据
    """
    print(f"生成演示数据...")
    
    # 生成训练数据
    train_data = []
    for sid in range(num_train_students):
        n_answers = random.randint(30, 50)
        qids = random.sample(range(num_questions), n_answers)
        for qid in qids:
            correct = random.choice([0, 1])
            train_data.append([sid, qid, correct])
    
    # 生成测试数据
    test_data = []
    for sid in range(num_test_students):
        n_answers = random.randint(20, 40)
        qids = random.sample(range(num_questions), n_answers)
        for qid in qids:
            correct = random.choice([0, 1])
            test_data.append([sid, qid, correct])
    
    # 生成概念映射（每道题随机关联1-2个知识点）
    concept_map = {}
    for qid in range(num_questions):
        n_concepts = random.randint(1, 2)
        concepts = random.sample(range(num_concepts), n_concepts)
        concept_map[qid] = concepts
    
    return train_data, test_data, concept_map

def main():
    """主函数"""
    logger = setuplogger()
    
    # ============================================
    # 第一步：配置参数
    # ============================================
    
    NUM_TRAIN_STUDENTS = 80
    NUM_TEST_STUDENTS = 20
    NUM_QUESTIONS = 50
    NUM_CONCEPTS = 10
    
    # ============================================
    # 第二步：生成数据
    # ============================================
    
    logger.info("生成演示数据...")
    train_data, test_data, concept_map = generate_demo_data(
        num_train_students=NUM_TRAIN_STUDENTS,
        num_test_students=NUM_TEST_STUDENTS,
        num_questions=NUM_QUESTIONS,
        num_concepts=NUM_CONCEPTS
    )
    
    logger.info(f"训练数据量: {len(train_data)}")
    logger.info(f"测试数据量: {len(test_data)}")
    
    # ============================================
    # 第三步：配置模型
    # ============================================
    
    config = {
        'learning_rate': 0.002,
        'batch_size': 256,
        'num_epochs': 20,
        'num_dim': 1,           # 1=一维IRT, >1=多维IRT
        'device': 'cpu',
        'betas': (0.9, 0.999),
        'policy': 'notbobcat',  # 不使用BOBCAT策略
    }
    
    logger.info("=" * 50)
    logger.info("开始训练 IRT 模型")
    logger.info("=" * 50)
    
    # ============================================
    # 第四步：创建训练数据集
    # ============================================
    
    train_dataset = TrainDataset(
        train_data,
        concept_map,
        NUM_TRAIN_STUDENTS,
        NUM_QUESTIONS,
        NUM_CONCEPTS
    )
    
    logger.info(f"训练数据集创建成功:")
    logger.info(f"  - 学生数: {train_dataset.num_students}")
    logger.info(f"  - 题目数: {train_dataset.num_questions}")
    logger.info(f"  - 知识点数: {train_dataset.num_concepts}")
    
    # ============================================
    # 第五步：创建并训练模型
    # ============================================
    
    model = IRTModel(**config)
    model.init_model(train_dataset)
    
    logger.info("模型初始化完成")
    logger.info("开始训练...")
    model.train(train_dataset, log_step=5)
    
    logger.info("训练完成！")
    
    # ============================================
    # 第六步：保存模型
    # ============================================
    
    os.makedirs('ckpt', exist_ok=True)
    model.adaptest_save('ckpt/irt_model.pt')
    logger.info("模型已保存到 ckpt/irt_model.pt")
    
    # ============================================
    # 第七步：自适应测试
    # ============================================
    
    logger.info("=" * 50)
    logger.info("开始自适应测试")
    logger.info("=" * 50)
    
    test_dataset = AdapTestDataset(
        test_data,
        concept_map,
        NUM_TEST_STUDENTS,
        NUM_QUESTIONS,
        NUM_CONCEPTS
    )
    
    model.adaptest_load('ckpt/irt_model.pt')
    
    strategy = MFIStrategy()
    logger.info(f"使用策略: {strategy.name}")
    
    test_length = 10
    
    for it in range(test_length):
        selected_questions = strategy.adaptest_select(model, test_dataset)
        
        for student_id, question_id in selected_questions.items():
            if question_id in test_dataset.data[student_id]:
                test_dataset.apply_selection(student_id, question_id)
        
        model.adaptest_update(test_dataset)
        results = model.evaluate(test_dataset)
        logger.info(f"Iter {it}: AUC={results['auc']:.4f}, Acc={results['acc']:.4f}")
    
    logger.info("=" * 50)
    logger.info("完成！")
    logger.info("=" * 50)

if __name__ == '__main__':
    main()
