"""
工具函数模块 - 包含神经网络相关的工具类

本模块提供了神经网络策略中使用的辅助工具，包括：
- StraightThrough：直通式策略更新器（用于BOBCAT算法）
- hard_sample：硬采样函数
- Actor：策略网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def hard_sample(logits, dim=-1):
    """
    硬采样函数 - 将软最大概率转换为one-hot向量
    
    用于强化学习中的直通估计器（Straight-Through Estimator）技巧，
    使得梯度可以反向传播但动作选择是确定性的。
    
    Args:
        logits: torch.Tensor - 网络输出的logits
        dim: int - 进行softmax的维度
        
    Returns:
        y_hard: torch.Tensor - one-hot形式的硬采样结果
        index: torch.Tensor - 采样到的索引
    """
    # 计算软最大概率
    y_soft = F.softmax(logits, dim=-1)
    # 获取概率最大的索引
    index = y_soft.max(dim, keepdim=True)[1]
    # 创建one-hot向量
    y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
    # 直通估计：返回 y_hard，但梯度流经 y_soft
    ret = y_hard - y_soft.detach() + y_soft
    return ret, index


class Actor(nn.Module):
    """
    策略网络（Actor）
    
    用于BOBCAT等算法中的选题策略学习。
    根据当前状态（已作答题目）和动作掩码（可选题目）输出动作概率。
    """
    
    def __init__(self, state_dim, action_dim, n_latent_var=256):
        """
        初始化策略网络
        
        Args:
            state_dim: int - 状态维度（等于题目数量）
            action_dim: int - 动作维度（等于题目数量）
            n_latent_var: int - 隐藏层维度
        """
        super().__init__()
        # 状态编码层
        self.obs_layer = nn.Linear(state_dim, n_latent_var)
        # 策略输出层：两层网络+Tanh激活
        self.actor_layer = nn.Sequential(
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim)
        )

    def forward(self, state, action_mask):
        """
        前向传播
        
        Args:
            state: torch.Tensor - 当前状态（已作答题目的对错情况）
            action_mask: torch.Tensor - 动作掩码（1表示可选，0表示不可选）
            
        Returns:
            actions: torch.Tensor - 选择的动作
        """
        # 编码当前状态
        hidden_state = self.obs_layer(state)
        # 计算动作 logits
        logits = self.actor_layer(hidden_state)
        # 应用动作掩码：将不可选动作的logits设为负无穷
        inf_mask = torch.clamp(torch.log(action_mask.float()),
                               min=torch.finfo(torch.float32).min)
        logits = logits + inf_mask
        # 选择动作（硬采样）
        actions = hard_sample(logits)
        return actions


class StraightThrough:
    """
    直通式策略更新器
    
    用于BOBCAT算法中训练策略网络（Actor）。
    实现了策略梯度更新逻辑。
    """
    
    def __init__(self, state_dim, action_dim, lr, config):
        """
        初始化策略更新器
        
        Args:
            state_dim: int - 状态维度
            action_dim: int - 动作维度
            lr: float - 学习率
            config: dict - 配置字典，包含device和betas参数
        """
        self.lr = lr
        device = config['device']
        self.betas = config['betas']
        # 创建策略网络
        self.policy = Actor(state_dim, action_dim).to(device)
        # 创建优化器
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, betas=self.betas)

    def update(self, loss):
        """
        更新策略网络
        
        根据损失函数更新策略网络参数。
        
        Args:
            loss: torch.Tensor - 策略损失
        """
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
