import sys
import os
import json
import datetime
import logging

import torch
import numpy as np
import pandas as pd

sys.path.append('..')
import pyat

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

dataset = 'assistment'
# 读数据:测试学生的三元组 + 题目-知识点映射 + 元信息
test_triplets = pd.read_csv(f'../datasets/{dataset}/test_triplets.csv', encoding='utf-8').to_records(index=False)
concept_map = json.load(open(f'../datasets/{dataset}/concept_map.json', 'r'))
concept_map = {int(k): v for k, v in concept_map.items()}
metadata = json.load(open(f'../datasets/{dataset}/metadata.json', 'r'))

test_data = pyat.AdapTestDataset(test_triplets, concept_map,
                                 metadata['num_test_students'], metadata['num_questions'], metadata['num_concepts'])

config = {
    'learning_rate': 0.0025,
    'batch_size': 2048,
    'num_epochs': 8,
    'num_dim': 1,     # IRT 能力/区分度的维度
    'device': 'cpu',
}

logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format='[%(asctime)s %(levelname)s] %(message)s',
)

test_length = 50   # 考试长度:每个学生共选 50 道题
now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

# 四种策略同台对比:随机 / Fisher 信息量(MFI) / 纯 EMC(质量模块消融) / 完整 MAAT
strategies = (
    pyat.RandomStrategy(),
    pyat.FisherStrategy(),
    pyat.ExpectedModelChangeStrategy(),
    pyat.MAATStrategy(n_candidates=10),
)

records = []
for strategy in strategies:
    # 每个策略前重置随机种子,保证 theta 初始化一致,对比才公平
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = pyat.IRTModel(**config)
    model.adaptest_init(test_data)
    model.adaptest_preload('../models/irt/checkpoint.pt')
    test_data.reset()
    results = pyat.AdapTestDriver.run(model, strategy, test_data, test_length, f'../results/{now}')
    for step, metrics in enumerate(results):
        records.append({'strategy': strategy.name, 'step': step, **metrics})
    # 每跑完一个策略就落盘一次,中途中断也不丢结果
    os.makedirs(f'../results/{now}', exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(f'../results/{now}/results.csv', index=False)

logging.info(f'all done, results saved to ../results/{now}/results.csv')
