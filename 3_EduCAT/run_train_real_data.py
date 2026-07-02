"""
EduCAT 训练脚本 - 使用真实格式数据

运行方式：
    cd EduCAT
    python run_train_real_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CAT.model import IRTModel
from CAT.strategy import MFIStrategy
from CAT.dataset import TrainDataset, AdapTestDataset

import json
import pandas as pd
import numpy as np

SEED = 42
np.random.seed(SEED)

def setuplogger():
    import logging
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    return root

def main():
    logger = setuplogger()
    
    # ============================================
    # 加载真实格式数据
    # ============================================
    
    data_dir = 'data/sample'
    
    logger.info("加载数据...")
    
    # 读取 CSV
    train_df = pd.read_csv(f'{data_dir}/train_triples.csv')
    test_df = pd.read_csv(f'{data_dir}/test_triples.csv')
    
    # 读取元数据
    metadata = json.load(open(f'{data_dir}/metadata.json'))
    concept_map = json.load(open(f'{data_dir}/concept_map.json'))
    
    # 转换为列表格式
    train_data = train_df.values.tolist()
    test_data = test_df.values.tolist()
    
    # 转换 concept_map 的 key
    concept_map = {int(k): v for k, v in concept_map.items()}
    
    logger.info(f"训练数据量: {len(train_data)}")
    logger.info(f"测试数据量: {len(test_data)}")
    logger.info(f"学生数: {metadata['num_students']}")
    logger.info(f"题目数: {metadata['num_questions']}")
    logger.info(f"知识点数: {metadata['num_concepts']}")
    
    # ============================================
    # 配置模型
    # ============================================
    
    config = {
        'learning_rate': 0.01,
        'batch_size': 128,
        'num_epochs': 50,
        'num_dim': 1,
        'device': 'cpu',
        'policy': 'notbobcat',
        'betas': (0.9, 0.999),
    }
    
    logger.info("=" * 50)
    logger.info("开始训练 IRT 模型")
    logger.info("=" * 50)
    
    # ============================================
    # 创建训练数据集
    # ============================================
    
    train_dataset = TrainDataset(
        train_data,
        concept_map,
        metadata['num_train_students'],
        metadata['num_questions'],
        metadata['num_concepts']
    )
    
    # ============================================
    # 创建并训练模型
    # ============================================
    
    model = IRTModel(**config)
    model.init_model(train_dataset)
    
    logger.info("开始训练...")
    model.train(train_dataset, log_step=10)
    
    logger.info("训练完成！")
    
    # 保存模型
    os.makedirs('ckpt', exist_ok=True)
    model.adaptest_save('ckpt/irt_model_real.pt')
    logger.info("模型已保存到 ckpt/irt_model_real.pt")
    
    # ============================================
    # 自适应测试
    # ============================================
    
    logger.info("=" * 50)
    logger.info("开始自适应测试")
    logger.info("=" * 50)
    
    test_dataset = AdapTestDataset(
        test_data,
        concept_map,
        metadata['num_test_students'],
        metadata['num_questions'],
        metadata['num_concepts']
    )
    
    model.adaptest_load('ckpt/irt_model_real.pt')
    
    strategy = MFIStrategy()
    
    for it in range(15):
        selected = strategy.adaptest_select(model, test_dataset)
        
        if not selected:
            logger.info(f"Iter {it}: 所有学生已完成测试")
            break
        
        for sid, qid in selected.items():
            if qid in test_dataset.data[sid]:
                test_dataset.apply_selection(sid, qid)
        
        model.adaptest_update(test_dataset)
        results = model.evaluate(test_dataset)
        logger.info(f"Iter {it}: AUC={results['auc']:.4f}, Acc={results['acc']:.4f}")
    
    logger.info("=" * 50)
    logger.info("完成！")
    logger.info("=" * 50)

if __name__ == '__main__':
    main()
