import sys
import json
import logging
import numpy as np
import pandas as pd

sys.path.append('..')
import pyat

dataset = 'assistment'

# 读数据:训练学生的三元组 + 题目-知识点映射 + 元信息
train_triplets = pd.read_csv(f'../datasets/{dataset}/train_triplets.csv', encoding='utf-8').to_records(index=False)
concept_map = json.load(open(f'../datasets/{dataset}/concept_map.json', 'r'))
concept_map = {int(k): v for k, v in concept_map.items()}
metadata = json.load(open(f'../datasets/{dataset}/metadata.json', 'r'))

# 构造训练数据集
train_data = pyat.TrainDataset(train_triplets, concept_map,
                               metadata['num_train_students'], metadata['num_questions'], metadata['num_concepts'])

config = {
    'learning_rate': 0.002,
    'batch_size': 2048,
    'num_epochs': 100,
    'num_dim': 1,     # IRT 能力/区分度的维度
    'device': 'cpu',
}

logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format='[%(asctime)s %(levelname)s] %(message)s',
)

# 离线预训练 IRT,保存题目参数 alpha/beta 供测试阶段加载
model = pyat.IRTModel(**config)
model.adaptest_init(train_data)
model.adaptest_train(train_data)
model.adaptest_save('../models/irt/checkpoint.pt')
