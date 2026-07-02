"""
项目反应理论（IRT）模型实现

IRT是一种经典的认知诊断模型，用于评估学生的能力水平和题目的难度。
本模块实现了：
- IRT：基础的IRT神经网络模型
- IRTModel：封装了训练、评估、保存等完整功能的IRT模型类

支持一维IRT和多维IRT（MIRT）。
"""

import vegas
import logging
import torch
import torch.nn as nn
import numpy as np
import math
import torch.utils.data as data
from math import exp as exp
from sklearn.metrics import roc_auc_score
from scipy import integrate
from CAT.model.abstract_model import AbstractModel
from CAT.dataset import AdapTestDataset, TrainDataset, Dataset
from sklearn.metrics import accuracy_score
from collections import namedtuple
from .utils import StraightThrough

# 用于强化学习中的动作保存
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class IRT(nn.Module):
    """
    项目反应理论（IRT）神经网络模型
    
    采用神经网络参数化的方式实现IRT模型：
    - theta: 学生能力向量
    - alpha: 题目区分度向量
    - beta: 题目难度标量
    
    答题概率公式: P(success) = sigmoid(alpha * theta + beta)
    """
    
    def __init__(self, num_students, num_questions, num_dim):
        """
        初始化IRT模型
        
        Args:
            num_students: int - 学生数量
            num_questions: int - 题目数量
            num_dim: int - 能力维度（1为IRT，>1为MIRT）
        """
        super().__init__()
        self.num_dim = num_dim  # 能力维度
        self.num_students = num_students
        self.num_questions = num_questions
        
        # 学生能力向量（可学习参数）
        self.theta = nn.Embedding(self.num_students, self.num_dim)
        # 题目区分度向量（可学习参数）
        self.alpha = nn.Embedding(self.num_questions, self.num_dim)
        # 题目难度标量（可学习参数）
        self.beta = nn.Embedding(self.num_questions, 1)

        # 使用Xavier初始化权重
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, student_ids, question_ids):
        """
        前向传播 - 计算答题正确概率
        
        Args:
            student_ids: torch.Tensor - 学生ID
            question_ids: torch.Tensor - 题目ID
            
        Returns:
            torch.Tensor - 答题正确的概率
        """
        # 获取学生能力和题目参数
        theta = self.theta(student_ids)  # (batch_size, num_dim)
        alpha = self.alpha(question_ids)  # (batch_size, num_dim)
        beta = self.beta(question_ids)  # (batch_size, 1)
        
        # 计算概率：sigmoid(alpha * theta + beta)
        pred = (alpha * theta).sum(dim=1, keepdim=True) + beta
        pred = torch.sigmoid(pred)
        return pred


class IRTModel(AbstractModel):
    """
    IRT模型类 - 封装了完整的IRT模型功能
    
    继承自AbstractModel，实现了：
    - 模型初始化和训练
    - 自适应测试更新
    - 模型评估（计算AUC、准确率、覆盖率）
    - 选题策略所需的信息计算（Fisher信息、KL信息等）
    """

    def __init__(self, **config):
        """
        初始化IRT模型类
        
        Args:
            config: dict - 配置字典，包含num_dim、learning_rate等参数
        """
        super().__init__()
        self.config = config
        self.model = None

    @property
    def name(self):
        return 'Item Response Theory'

    def init_model(self, data: Dataset):
        """
        初始化IRT模型
        
        Args:
            data: Dataset - 数据集对象
        """
        policy_lr = 0.0005
        # 创建IRT模型
        self.model = IRT(data.num_students, data.num_questions, self.config['num_dim'])
        # 创建策略网络（用于BOBCAT算法）
        self.policy = StraightThrough(data.num_questions, data.num_questions, policy_lr, self.config)
        self.n_q = data.num_questions

    def train(self, train_data: TrainDataset, log_step=1):
        """
        训练IRT模型
        
        使用学生-题目-答案三元组数据训练模型，学习题目参数和学生能力。
        
        Args:
            train_data: TrainDataset - 训练数据集
            log_step: int - 日志打印间隔
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
            for cnt, (student_ids, question_ids, _, labels) in enumerate(train_loader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                labels = labels.to(device).float()
                pred = self.model(student_ids, question_ids).view(-1)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                loss += bz_loss.data.float()
                if cnt % log_step == 0:
                    logging.info('Epoch [{}] Batch [{}]: loss={:.5f}'.format(ep, cnt, loss / cnt))

    def adaptest_save(self, path):
        """
        保存模型 - 只保存题目参数（alpha, beta）
        
        自适应测试中只保存题目相关参数，学生能力是动态估计的。
        
        Args:
            path: str - 保存路径
        """
        model_dict = self.model.state_dict()
        model_dict = {k: v for k, v in model_dict.items() if 'alpha' in k or 'beta' in k}
        torch.save(model_dict, path)

    def adaptest_load(self, path):
        """
        加载模型
        
        Args:
            path: str - 模型路径
        """
        # BOBCAT算法需要加载策略网络
        if self.config['policy'] == 'bobcat':
            self.policy.policy.load_state_dict(torch.load(self.config['policy_path']), strict=False)
        self.model.load_state_dict(torch.load(path), strict=False)
        self.model.to(self.config['device'])

    def adaptest_update(self, adaptest_data: AdapTestDataset, sid=None):
        """
        更新模型 - 使用测试数据更新学生能力估计
        
        在自适应测试过程中，根据学生已作答的题目更新其能力估计。
        
        Args:
            adaptest_data: AdapTestDataset - 测试数据集
            sid: int - 学生ID（可选，用于指定单个学生）
        """
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        # 只更新学生能力theta，不更新题目参数
        optimizer = torch.optim.Adam(self.model.theta.parameters(), lr=lr)

        tested_dataset = adaptest_data.get_tested_dataset(last=True, ssid=sid)
        dataloader = torch.utils.data.DataLoader(tested_dataset, batch_size=batch_size, shuffle=True)
        
        for ep in range(1, epochs + 1):
            loss = 0.0
            for cnt, (student_ids, question_ids, _, labels) in enumerate(dataloader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                labels = labels.to(device).float()
                pred = self.model(student_ids, question_ids).view(-1)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                loss += bz_loss.data.float()
        return loss

    def one_student_update(self, adaptest_data: AdapTestDataset):
        """
        单学生更新 - 为单个学生更新能力估计
        """
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        optimizer = torch.optim.Adam(self.model.theta.parameters(), lr=lr)

    def evaluate(self, adaptest_data: AdapTestDataset):
        """
        评估模型性能
        
        计算AUC、准确率和知识点覆盖率。
        
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
                real += [data[sid][qid] for qid in question_ids]
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                output = self.model(student_ids, question_ids).view(-1)
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
        pred = np.array(pred)
        # 计算AUC
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
        获取模型预测 - 返回所有学生-题目对的预测概率
        
        Args:
            adaptest_data: AdapTestDataset - 测试数据集
            
        Returns:
            dict - {学生ID: {题目ID: 预测概率}}
        """
        data = adaptest_data.data
        device = self.config['device']

        pred_all = {}

        with torch.no_grad():
            self.model.eval()
            for sid in data:
                pred_all[sid] = {}
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                output = self.model(student_ids, question_ids).view(-1).tolist()
                for i, qid in enumerate(list(data[sid].keys())):
                    pred_all[sid][qid] = output[i]
            self.model.train()

        return pred_all

    def _loss_function(self, pred, real):
        """
        二元交叉熵损失函数
        
        Args:
            pred: torch.Tensor - 预测概率
            real: torch.Tensor - 真实标签
            
        Returns:
            torch.Tensor - 损失值
        """
        return -(real * torch.log(0.0001 + pred) + (1 - real) * torch.log(1.0001 - pred)).mean()
    
    def get_alpha(self, question_id):
        """
        获取题目的区分度参数
        
        Args:
            question_id: int - 题目ID
            
        Returns:
            np.ndarray - 区分度向量，形状为 (num_dim,)
        """
        return self.model.alpha.weight.data.cpu().numpy()[question_id]
    
    def get_beta(self, question_id):
        """
        获取题目的难度参数
        
        Args:
            question_id: int - 题目ID
            
        Returns:
            np.ndarray - 难度值
        """
        return self.model.beta.weight.data.cpu().numpy()[question_id]
    
    def get_theta(self, student_id):
        """
        获取学生的能力参数
        
        Args:
            student_id: int - 学生ID
            
        Returns:
            np.ndarray - 能力向量，形状为 (num_dim,)
        """
        return self.model.theta.weight.data.cpu().numpy()[student_id]

    def get_kli(self, student_id, question_id, n, pred_all):
        """
        计算KL信息量（Kullback-Leibler Information）
        
        用于KLI选题策略，选择能够提供最多关于学生能力信息的题目。
        
        Args:
            student_id: int - 学生ID
            question_id: int - 题目ID
            n: int - 当前已作答题目数量
            pred_all: dict - 所有预测概率
            
        Returns:
            float - KL信息量
        """
        if n == 0:
            return np.inf
        device = self.config['device']
        dim = self.model.num_dim
        sid = torch.LongTensor([student_id]).to(device)
        qid = torch.LongTensor([question_id]).to(device)
        theta = self.get_theta(sid)  # (num_dim,)
        alpha = self.get_alpha(qid)  # (num_dim,)
        beta = self.get_beta(qid)[0]  # float
        pred_estimate = pred_all[student_id][question_id]
        
        def kli(x):
            """KL信息公式，用于数值积分"""
            if type(x) == float:
                x = np.array([x])
            pred = np.matmul(alpha.T, x) + beta
            pred = 1 / (1 + np.exp(-pred))
            q_estimate = 1 - pred_estimate
            q = 1 - pred
            return pred_estimate * np.log(pred_estimate / pred) + q_estimate * np.log((q_estimate / q))
        
        # 设置积分边界（基于置信区间）
        c = 3
        boundaries = [[theta[i] - c / np.sqrt(n), theta[i] + c / np.sqrt(n)] for i in range(dim)]
        
        if len(boundaries) == 1:
            # KLI（一维情况）
            v, err = integrate.quad(kli, boundaries[0][0], boundaries[0][1])
            return v
        # MKLI（多维情况）
        integ = vegas.Integrator(boundaries)
        result = integ(kli, nitn=10, neval=1000)
        return result.mean

    def get_fisher(self, student_id, question_id, pred_all):
        """
        计算Fisher信息量
        
        用于MFI选题策略，选择能够最大程度减少能力估计方差的题目。
        
        Args:
            student_id: int - 学生ID
            question_id: int - 题目ID
            pred_all: dict - 所有预测概率
            
        Returns:
            np.ndarray - Fisher信息矩阵，形状为 (num_dim, num_dim)
        """
        device = self.config['device']
        qid = torch.LongTensor([question_id]).to(device)
        alpha = self.model.alpha(qid).clone().detach().cpu()
        pred = pred_all[student_id][question_id]
        q = 1 - pred
        fisher_info = (q * pred * (alpha * alpha.T)).numpy()
        return fisher_info
    
    def bce_loss_derivative(self, pred, target):
        """
        计算BCE损失的导数
        
        用于BECAT算法中的边界估计。
        
        Args:
            pred: float - 预测概率
            target: int - 目标标签
            
        Returns:
            float - 损失函数导数
        """
        derivative = (pred - target) / (pred * (1 - pred))
        return derivative
    
    def get_BE_weights(self, pred_all):
        """
        计算BECAT算法的边界估计权重
        
        用于选择能够最有效缩小能力估计边界的题目。
        
        Args:
            pred_all: dict - 题目预测概率
            
        Returns:
            dict - 权重矩阵
        """
        d = 100
        Pre_true = {}
        Pre_false = {}
        Der = {}
        for qid, pred in pred_all.items():
            Pre_true[qid] = pred
            Pre_false[qid] = 1 - pred
            Der[qid] = pred * (1 - pred) * self.get_alpha(qid)
        
        w_ij_matrix = {}
        for i, _ in pred_all.items():
            w_ij_matrix[i] = {}
            for j, _ in pred_all.items():
                w_ij_matrix[i][j] = 0
        
        for i, _ in pred_all.items():
            for j, _ in pred_all.items():
                gradients_theta1 = self.bce_loss_derivative(Pre_true[i], 1.0) * Der[i]
                gradients_theta2 = self.bce_loss_derivative(Pre_true[i], 0.0) * Der[i]
                gradients_theta3 = self.bce_loss_derivative(Pre_true[j], 1.0) * Der[j]
                gradients_theta4 = self.bce_loss_derivative(Pre_true[j], 0.0) * Der[j]
                diff_norm_00 = math.fabs(gradients_theta1 - gradients_theta3)
                diff_norm_01 = math.fabs(gradients_theta1 - gradients_theta4)
                diff_norm_10 = math.fabs(gradients_theta2 - gradients_theta3)
                diff_norm_11 = math.fabs(gradients_theta2 - gradients_theta4)
                Expect = (Pre_false[i] * Pre_false[j] * diff_norm_00 + 
                         Pre_false[i] * Pre_true[j] * diff_norm_01 +
                         Pre_true[i] * Pre_false[j] * diff_norm_10 + 
                         Pre_true[i] * Pre_true[j] * diff_norm_11)
                w_ij_matrix[i][j] = d - Expect
        return w_ij_matrix

    def F_s_func(self, S_set, w_ij_matrix):
        """
        计算已选题目集的F_s分数
        
        用于评估一组已选题目对能力边界的影响。
        
        Args:
            S_set: list - 已选题目ID列表
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
        
        评估将某题目加入当前已选集后F_s的变化。
        
        Args:
            question_id: int - 待评估题目ID
            pred_all: dict - 预测概率
            S_set: list - 当前已选题目集
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

    def expected_model_change(self, sid: int, qid: int, adaptest_data: AdapTestDataset, pred_all: dict):
        """
        计算预期模型变化（Expected Model Change）
        
        用于MAAT选题策略，评估选择某题目后模型参数的变化程度。
        
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

        # 冻结非theta参数
        for name, param in self.model.named_parameters():
            if 'theta' not in name:
                param.requires_grad = False

        original_weights = self.model.theta.weight.data.clone()

        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        correct = torch.LongTensor([1]).to(device).float()
        wrong = torch.LongTensor([0]).to(device).float()

        # 计算答对时的参数变化
        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id)
            loss = self._loss_function(pred, correct)
            loss.backward()
            optimizer.step()

        pos_weights = self.model.theta.weight.data.clone()
        self.model.theta.weight.data.copy_(original_weights)

        # 计算答错时的参数变化
        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id)
            loss = self._loss_function(pred, wrong)
            loss.backward()
            optimizer.step()

        neg_weights = self.model.theta.weight.data.clone()
        self.model.theta.weight.data.copy_(original_weights)

        # 恢复所有参数的可训练状态
        for param in self.model.parameters():
            param.requires_grad = True

        pred = pred_all[sid][qid]
        # 加权求和：P(答对) * 变化量 + P(答错) * 变化量
        return pred * torch.norm(pos_weights - original_weights).item() + \
               (1 - pred) * torch.norm(neg_weights - original_weights).item()
    
    def bobcat_policy(self, S_set, untested_questions):
        """
        BOBCAT策略的选题函数
        
        使用训练好的策略网络选择下一道题目。
        
        Args:
            S_set: list - 已选题目的状态字典列表
            untested_questions: dict - 未测试题目ID集合
            
        Returns:
            int - 选择的题目ID
        """
        device = self.config['device']
        action_mask = [0.0] * self.n_q
        train_mask = [-0.0] * self.n_q
        
        # 设置动作掩码
        for index in untested_questions:
            action_mask[index] = 1.0
        
        # 设置训练掩码（历史作答情况）
        for state in S_set:
            keys = list(state.keys())
            key = keys[0]
            values = list(state.values())
            val = values[0]
            train_mask[key] = (float(val) - 0.5) * 2
        
        action_mask = torch.tensor(action_mask).to(device)
        train_mask = torch.tensor(train_mask).to(device)
        action = self.policy.policy(train_mask, action_mask)
        return action.item()
