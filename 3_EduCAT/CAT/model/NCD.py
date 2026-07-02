"""
神经认知诊断（NCD）模型实现

NCD是一种基于深度学习的认知诊断框架，用于处理学生与题目之间复杂的非线性交互关系。
本模块实现了：
- NCD：基础的神经认知诊断神经网络模型
- NCDModel：封装了训练、评估、保存等完整功能的NCD模型类

特点：
- 使用嵌入层表示学生和题目
- 使用全连接神经网络学习复杂的答题交互
- 支持知识点关联的题目难度建模
"""

import torch
import logging
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import roc_auc_score, accuracy_score
import math
from CAT.model.abstract_model import AbstractModel
from CAT.dataset import AdapTestDataset, TrainDataset, Dataset


class NCD(nn.Module):
    """
    神经认知诊断模型（Neural Cognitive Diagnosis）
    
    使用神经网络来建模学生能力、题目难度和知识点之间的复杂关系。
    
    网络结构：
    1. 学生嵌入层 -> 学生能力向量
    2. 题目难度嵌入层 -> 知识点难度向量
    3. 题目区分度嵌入层 -> 区分度标量
    4. 知识相关性向量（外部提供）
    5. 预测网络 -> 输出答题正确概率
    """
    
    def __init__(self, student_n, exer_n, knowledge_n, prednet_len1=128, prednet_len2=64):
        """
        初始化NCD模型
        
        Args:
            student_n: int - 学生数量
            exer_n: int - 题目数量
            knowledge_n: int - 知识点数量
            prednet_len1: int - 第一层隐藏层维度（默认128）
            prednet_len2: int - 第二层隐藏层维度（默认64）
        """
        self.knowledge_dim = knowledge_n  # 知识点维度
        self.exer_n = exer_n  # 题目数量
        self.emb_num = student_n  # 学生嵌入数量
        self.stu_dim = self.knowledge_dim  # 学生嵌入维度等于知识点数量
        self.prednet_input_len = self.knowledge_dim  # 预测网络输入维度
        self.prednet_len1, self.prednet_len2 = prednet_len1, prednet_len2

        super(NCD, self).__init__()

        # 网络结构定义
        # 学生嵌入：学习学生的知识点掌握情况
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        # 知识点难度嵌入：每个题目涉及的各知识点难度
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        # 题目区分度嵌入：题目的整体难度/区分度
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        # 预测网络：三层全连接网络
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)  # Dropout防止过拟合
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # Xavier初始化
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb):
        """
        前向传播 - 计算答题正确概率
        
        Args:
            stu_id: torch.LongTensor - 学生ID
            exer_id: torch.LongTensor - 题目ID
            kn_emb: torch.FloatTensor - 知识点相关性向量
            
        Returns:
            torch.FloatTensor - 答题正确的概率
        """
        # 获取学生能力向量
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        # 获取题目各知识点难度
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        # 获取题目整体区分度
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        
        # 构建预测网络输入
        # 公式：区分度 * (学生能力 - 题目难度) * 知识点相关性
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        # 通过全连接网络
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))

        return output

    def apply_clipper(self):
        """
        应用梯度裁剪 - 限制参数为非负数
        
        确保难度和区分度参数为正数，符合实际意义。
        """
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        """
        获取学生的知识状态（知识点掌握向量）
        
        Args:
            stu_id: int - 学生ID
            
        Returns:
            torch.Tensor - 知识点掌握向量
        """
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        """
        获取题目的参数
        
        Args:
            exer_id: int - 题目ID
            
        Returns:
            tuple - (知识点难度向量, 题目区分度)
        """
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data


class NoneNegClipper(object):
    """
    非负裁剪器
    
    用于将网络参数限制为非负值。
    适用于难度、区分度等应该为正数的参数。
    """
    
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        """对模块的参数应用非负裁剪"""
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))  # 获取负值部分
            w.add_(a)  # 将负值变为0


class NCDModel(AbstractModel):
    """
    NCD模型类 - 封装了完整的神经认知诊断模型功能
    
    继承自AbstractModel，实现了：
    - 模型初始化和训练
    - 自适应测试更新
    - 模型评估
    - 选题策略所需的信息计算
    """

    def __init__(self, **config):
        """
        初始化NCD模型类
        
        Args:
            config: dict - 配置字典，包含prednet_len1、prednet_len2等参数
        """
        super().__init__()
        self.config = config
        self.model = None

    @property
    def name(self):
        return 'Neural Cognitive Diagnosis'

    def init_model(self, data: Dataset):
        """
        初始化NCD模型
        
        Args:
            data: Dataset - 数据集对象
        """
        self.model = NCD(
            data.num_students, 
            data.num_questions, 
            data.num_concepts, 
            self.config['prednet_len1'], 
            self.config['prednet_len2']
        )
    
    def train(self, train_data: TrainDataset):
        """
        训练NCD模型
        
        使用学生-题目-知识点-答案数据训练模型。
        
        Args:
            train_data: TrainDataset - 训练数据集
        """
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        self.model.to(device)
        logging.info('train on {}'.format(device))

        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for ep in range(1, epochs + 1):
            loss = 0.0
            log_step = 1
            for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(train_loader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                concepts_emb = concepts_emb.to(device)
                labels = labels.to(device)
                pred = self.model(student_ids, question_ids, concepts_emb)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                self.model.apply_clipper()  # 应用参数裁剪
                loss += bz_loss.data.float()
                if cnt % log_step == 0:
                    logging.info('Epoch [{}] Batch [{}]: loss={:.5f}'.format(ep, cnt, loss / cnt))
    
    def _loss_function(self, pred, real):
        """
        负对数似然损失函数
        
        Args:
            pred: torch.Tensor - 预测概率
            real: torch.Tensor - 真实标签
            
        Returns:
            torch.Tensor - 损失值
        """
        pred_0 = torch.ones(pred.size()).to(self.config['device']) - pred
        output = torch.cat((pred_0, pred), 1)
        criteria = nn.NLLLoss()
        return criteria(torch.log(output), real)
    
    def adaptest_save(self, path):
        """
        保存模型 - 不保存学生参数
        
        Args:
            path: str - 保存路径
        """
        model_dict = self.model.state_dict()
        # 过滤掉学生相关的参数
        model_dict = {k: v for k, v in model_dict.items() if 'student' not in k}
        torch.save(model_dict, path)
    
    def adaptest_load(self, path):
        """
        加载模型
        
        Args:
            path: str - 模型路径
        """
        self.model.load_state_dict(torch.load(path), strict=False)
        self.model.to(self.config['device'])
    
    def adaptest_update(self, adaptest_data: AdapTestDataset):
        """
        更新模型 - 使用测试数据更新学生能力估计
        
        Args:
            adaptest_data: AdapTestDataset - 测试数据集
        """
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        # 只更新学生嵌入参数
        optimizer = torch.optim.Adam(self.model.student_emb.parameters(), lr=lr)

        tested_dataset = adaptest_data.get_tested_dataset(last=True)
        dataloader = torch.utils.data.DataLoader(tested_dataset, batch_size=batch_size, shuffle=True)

        for ep in range(1, epochs + 1):
            loss = 0.0
            for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(dataloader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                labels = labels.to(device)
                concepts_emb = concepts_emb.to(device)
                pred = self.model(student_ids, question_ids, concepts_emb)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                self.model.apply_clipper()
                loss += bz_loss.data.float()

    def evaluate(self, adaptest_data: AdapTestDataset):
        """
        评估模型性能
        
        Args:
            adaptest_data: AdapTestDataset - 测试数据集
            
        Returns:
            dict - {'auc': AUC值, 'cov': 知识点覆盖率, 'acc': 准确率}
        """
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
        device = self.config['device']

        real = []
        pred = []
        with torch.no_grad():
            self.model.eval()
            for sid in data:
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                # 构建知识点嵌入
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                real += [data[sid][qid] for qid in question_ids]
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_embs = torch.Tensor(concepts_embs).to(device)
                output = self.model(student_ids, question_ids, concepts_embs).view(-1)
                pred += output.tolist()
            self.model.train()

        # 计算知识点覆盖率
        coverages = []
        for sid in data:
            all_concepts = set()
            tested_concepts = set()
            for qid in data[sid]:
                all_concepts.update(set(concept_map[qid]))
            for qid in adaptest_data.tested[sid]:
                tested_concepts.update(set(concept_map[qid]))
            coverage = len(tested_concepts) / len(all_concepts)
            coverages.append(coverage)
        cov = sum(coverages) / len(coverages)

        real = np.array(real)
        real = np.where(real < 0.5, 0.0, 1.0)  # 转换为二值标签
        pred = np.array(pred)
        auc = roc_auc_score(real, pred)
        
        # 计算准确率
        threshold = 0.5
        binary_pred = (pred >= threshold).astype(int)
        acc = accuracy_score(real, binary_pred)

        return {
            'auc': auc,
            'cov': cov,
            'acc': acc
        }
    
    def get_pred(self, adaptest_data: AdapTestDataset):
        """
        获取模型预测
        
        Args:
            adaptest_data: AdapTestDataset - 测试数据集
            
        Returns:
            dict - {学生ID: {题目ID: 预测概率}}
        """
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
        device = self.config['device']

        pred_all = {}
        with torch.no_grad():
            self.model.eval()
            for sid in data:
                pred_all[sid] = {}
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_embs = torch.Tensor(concepts_embs).to(device)
                output = self.model(student_ids, question_ids, concepts_embs).view(-1).tolist()
                for i, qid in enumerate(list(data[sid].keys())):
                    pred_all[sid][qid] = output[i]
            self.model.train()
        return pred_all

    def expected_model_change(self, sid: int, qid: int, adaptest_data: AdapTestDataset, pred_all: dict):
        """
        计算预期模型变化（Expected Model Change）
        
        用于MAAT选题策略。
        
        Args:
            sid: int - 学生ID
            qid: int - 题目ID
            adaptest_data: AdapTestDataset - 测试数据
            pred_all: dict - 预测概率
            
        Returns:
            float - 预期模型变化量
        """
        epochs = self.config['num_epochs']
        lr = self.config['learning_rate']
        device = self.config['device']
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # 冻结非学生参数
        for name, param in self.model.named_parameters():
            if 'student' not in name:
                param.requires_grad = False

        original_weights = self.model.student_emb.weight.data.clone()

        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        concepts = adaptest_data.concept_map[qid]
        concepts_emb = [0.] * adaptest_data.num_concepts
        for concept in concepts:
            concepts_emb[concept] = 1.0
        concepts_emb = torch.Tensor([concepts_emb]).to(device)
        correct = torch.LongTensor([1]).to(device)
        wrong = torch.LongTensor([0]).to(device)

        # 计算答对时的参数变化
        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id, concepts_emb)
            loss = self._loss_function(pred, correct)
            loss.backward()
            optimizer.step()

        pos_weights = self.model.student_emb.weight.data.clone()
        self.model.student_emb.weight.data.copy_(original_weights)

        # 计算答错时的参数变化
        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id, concepts_emb)
            loss = self._loss_function(pred, wrong)
            loss.backward()
            optimizer.step()

        neg_weights = self.model.student_emb.weight.data.clone()
        self.model.student_emb.weight.data.copy_(original_weights)

        # 恢复所有参数
        for param in self.model.parameters():
            param.requires_grad = True

        pred = pred_all[sid][qid]
        return pred * torch.norm(pos_weights - original_weights).item() + \
               (1 - pred) * torch.norm(neg_weights - original_weights).item()
               
    def get_BE_weights(self, pred_all):
        """
        计算BECAT算法的边界估计权重
        
        Args:
            pred_all: dict - 预测概率
            
        Returns:
            dict - 权重矩阵
        """
        d = 100
        Pre_true = {}
        Pre_false = {}
        for qid, pred in pred_all.items():
            Pre_true[qid] = pred
            Pre_false[qid] = 1 - pred
        
        w_ij_matrix = {}
        for i, _ in pred_all.items():
            w_ij_matrix[i] = {}
            for j, _ in pred_all.items():
                w_ij_matrix[i][j] = 0
        
        for i, _ in pred_all.items():
            for j, _ in pred_all.items():
                criterion_true_1 = nn.BCELoss()
                criterion_false_1 = nn.BCELoss()
                criterion_true_0 = nn.BCELoss()
                criterion_false_0 = nn.BCELoss()
                tensor_11 = torch.tensor(Pre_true[i], requires_grad=True)
                tensor_12 = torch.tensor(Pre_true[j], requires_grad=True)
                loss_true_1 = criterion_true_1(tensor_11, torch.tensor(1.0))
                loss_false_1 = criterion_false_1(tensor_11, torch.tensor(0.0))
                loss_true_0 = criterion_true_0(tensor_12, torch.tensor(1.0))
                loss_false_0 = criterion_false_0(tensor_12, torch.tensor(0.0))
                loss_true_1.backward()
                grad_true_1 = tensor_11.grad.clone()
                tensor_11.grad.zero_()
                loss_false_1.backward()
                grad_false_1 = tensor_11.grad.clone()
                tensor_11.grad.zero_()
                loss_true_0.backward()
                grad_true_0 = tensor_12.grad.clone()
                tensor_12.grad.zero_()
                loss_false_0.backward()
                grad_false_0 = tensor_12.grad.clone()
                tensor_12.grad.zero_()
                diff_norm_00 = math.fabs(grad_true_1 - grad_true_0)
                diff_norm_01 = math.fabs(grad_true_1 - grad_false_0)
                diff_norm_10 = math.fabs(grad_false_1 - grad_true_0)
                diff_norm_11 = math.fabs(grad_false_1 - grad_false_0)
                Expect = (Pre_false[i] * Pre_false[j] * diff_norm_00 + 
                         Pre_false[i] * Pre_true[j] * diff_norm_01 +
                         Pre_true[i] * Pre_false[j] * diff_norm_10 + 
                         Pre_true[i] * Pre_true[j] * diff_norm_11)
                w_ij_matrix[i][j] = d - Expect
        return w_ij_matrix

    def F_s_func(self, S_set, w_ij_matrix):
        """
        计算已选题目集的F_s分数
        
        Args:
            S_set: list - 已选题目集
            w_ij_matrix: dict - 权重矩阵
            
        Returns:
            float - F_s分数
        """
        res = 0.0
        for w_i in w_ij_matrix:
            if w_i not in S_set:
                mx = float('-inf')
                for j in S_set:
                    if w_ij_matrix[w_i][j] > mx:
                        mx = w_ij_matrix[w_i][j]
                res += mx
        return res

    def delta_q_S_t(self, question_id, pred_all, S_set, sampled_elements):
        """
        计算BECAT中题目的权重增量
        
        Args:
            question_id: int - 待评估题目ID
            pred_all: dict - 预测概率
            S_set: list - 已选题目集
            sampled_elements: np.ndarray - 采样元素
            
        Returns:
            float - F_sp - F_s 的差值
        """
        Sp_set = list(S_set)
        b_array = np.array(Sp_set)
        sampled_elements = np.concatenate((sampled_elements, b_array), axis=0)
        if question_id not in sampled_elements:
            sampled_elements = np.append(sampled_elements, question_id)
        sampled_dict = {key: value for key, value in pred_all.items() if key in sampled_elements}
        
        w_ij_matrix = self.get_BE_weights(sampled_dict)
        F_s = self.F_s_func(Sp_set, w_ij_matrix)
        
        Sp_set.append(question_id)
        F_sp = self.F_s_func(Sp_set, w_ij_matrix)
        return F_sp - F_s
