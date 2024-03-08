import time

import torch
import torch.nn as nn
import numpy as np
from utils import *
from network import Flashback
from scipy.sparse import csr_matrix


def maksed_mse_loss(input, target, mask_value=-1):
    mask = target == mask_value
    out = (input[~mask] - target[~mask]) ** 2
    loss = out.mean()
    return loss

def trajectory_forecasting_loss(pred, true):
    # L2 损失，即均方误差
    return F.mse_loss(pred, true, reduction='mean')

def consistency_loss(pred_aux, pred_main):
    # 根据 Eq. (14) 实现的一致性损失，其中 pred_aux 是辅助任务的预测，pred_main 是主任务的预测
    return F.mse_loss(pred_aux, pred_main, reduction='mean')

class FlashbackTrainer():
    """ Instantiates Flashback module with spatial and temporal weight functions.
    Performs loss computation and prediction.
    """

    def __init__(self, lambda_t, lambda_s, lambda_loc, lambda_user, use_weight, transition_graph, spatial_graph,
                 friend_graph, use_graph_user, use_spatial_graph, interact_graph):
        """ The hyper parameters to control spatial and temporal decay.
        """
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s

        self.lambda_loc = lambda_loc
        self.lambda_user = lambda_user
        self.use_weight = use_weight
        self.use_graph_user = use_graph_user
        self.use_spatial_graph = use_spatial_graph
        self.graph = transition_graph
        self.spatial_graph = spatial_graph
        self.friend_graph = friend_graph
        self.interact_graph = interact_graph
        # self.loss_weight = nn.Parameter(torch.ones(2)) 
        self.loss_weight = nn.Parameter(torch.ones(3))  

    def __str__(self):
        return 'Use flashback training.'

    def count_parameters(self):
        param_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_count += param.numel()
        return param_count
    
    def parameters(self):
        return self.model.parameters()

    def prepare(self, loc_count, user_count, hidden_size, gru_factory, device, setting):
        def f_t(delta_t, user_len): return ((torch.cos(delta_t * 2 * np.pi / 86400) + 1) / 2) * torch.exp(
            -(delta_t / 86400 * self.lambda_t))  # hover cosine + exp decay

        # exp decay  2个functions
        def f_s(delta_s, user_len): return torch.exp(-(delta_s * self.lambda_s))
        self.loc_count = loc_count
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.model = Flashback(loc_count, user_count, hidden_size, f_t, f_s, gru_factory, self.lambda_loc,
                               self.lambda_user, self.use_weight, self.graph, self.spatial_graph, self.friend_graph,
                               self.use_graph_user, self.use_spatial_graph, self.interact_graph, setting).to(device)

    def evaluate(self, x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_users, f, y_f, dataset):
        """ takes a batch (users x location sequence)
        then does the prediction and returns a list of user x sequence x location
        describing the probabilities for each location at each position in the sequence.
        t, s are temporal and spatial data related to the location sequence x
        y_t, y_s are temporal and spatial data related to the target sequence y.
        Flashback does not access y_t and y_s for prediction!
        """

        self.model.eval()
        # (seq_len, user_len, loc_count)
        out, h, _, _ = self.model(x, t, t_slot, s, y_t,
                            y_t_slot, y_s, h, active_users, f, y_f, dataset)

        out_t = out.transpose(0, 1)

        return out_t, h  # model outputs logits

    def loss(self, x, t, t_slot, s, y, y_t, y_t_slot, y_s, h, active_users, f, y_f, logits, dataset):
        """ takes a batch (users x location sequence)
        and corresponding targets in order to compute the training loss """

        self.model.train()
        out, h, cos, out_time = self.model(x, t, t_slot, s, y_t, y_t_slot, y_s, h,
                            active_users, f, y_f, dataset)  # out (seq_len, batch_size, loc_count)
        
        out = out.view(-1, self.loc_count)  # (seq_len * batch_size, loc_count)

        y = y.view(-1)  # (seq_len * batch_size)

        cos = cos.view(-1, self.loc_count)

        # 获取每个样本的真实类别对应的余弦值
        target_cosine = cos.gather(1, y.unsqueeze(1)).view(-1)

        vector_lengths = torch.where(target_cosine > 0, torch.ones_like(target_cosine), 1 - target_cosine)

        # 使用对数和来计算几何平均长度
        log_geom_mean_length = torch.log(vector_lengths + 1e-9).mean()
        geom_mean_length = torch.exp(log_geom_mean_length)

        # 计算长度与几何平均长度之间的差异，用于调整权重
        length_diff = vector_lengths - geom_mean_length
        
        # 根据长度差异调整权重
        weights = torch.ones_like(vector_lengths)
        weights[length_diff > 0] = 1 + length_diff[length_diff > 0]

        # 将权重应用于交叉熵损失
        l1 = (self.cross_entropy_loss(out+logits, y) * weights).mean()

        # 应用softmax来获得两个损失的权重
        loss_weights = F.softmax(self.loss_weight, dim=0)

        l2 = self.cross_entropy_loss(out, y)

        l3 = maksed_mse_loss(out_time.squeeze(-1), y_t_slot.squeeze(0)/168)


        l = loss_weights[0] * l1 + loss_weights[1] * l2 + loss_weights[2] * l3

        return l
