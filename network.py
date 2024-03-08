import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
from torch.nn import init
from enum import Enum
import time
import numpy as np
from utils import *
import scipy.sparse as sp
import math
from model import TransformerModel, EncoderLayer, Time2Vec, FuseEmbeddings, FuseEmbeddings1, Decoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import math

# 距离权重函数
def gaussian_distance_weight(haversine_distance, sigma=1):
    # sigma是高斯分布的标准差，决定了曲线的宽度
    # 我们假设距离的均值为0
    return torch.exp(-haversine_distance**2 / (2 * sigma**2))

# 转移次数权重函数
def migration_weight(migration_count, delta=1):
    # 使用对数来增加小数量迁移的影响力，delta避免log(0)
    return torch.log(1 + migration_count) / (1 + delta)

# 综合权重函数
def combined_weight(haversine_distance, migration_count, alpha=1, delta=1, method='add'):
    w_dist = gaussian_distance_weight(haversine_distance, alpha)
    w_migr = migration_weight(migration_count, delta)
    
    if method == 'add':
        return w_dist + w_migr
    elif method == 'mult':
        return w_dist * w_migr
    else:
        raise ValueError("Method must be 'add' or 'multiply'")

def haversine(s1, s2):
    """
    计算两批地理位置之间的Haversine距离。

    参数:
    s1, s2: 两批地理位置的张量，尺寸均为[batch_size, 2]，包含纬度和经度。

    返回:
    两批地理位置之间距离的张量，尺寸为[batch_size]。
    """

    # 将纬度和经度从度转换为弧度
    s1 = s1 * math.pi / 180
    s2 = s2 * math.pi / 180

    # 分离纬度和经度
    lat1, lon1 = s1[:, 0], s1[:, 1]
    lat2, lon2 = s2[:, 0], s2[:, 1]

    # Haversine公式
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a))

    # 地球平均半径，单位千米
    r = 6371

    return c * r

def haversines(s1, s2):
    """
    计算两批地理位置之间的Haversine距离。

    参数:
    s1, s2: 两批地理位置的张量，尺寸均为[batch_size, seq_len, 2]，包含纬度和经度。

    返回:
    两批地理位置之间距离的张量，尺寸为[batch_size, seq_len, seq_len]。
    """

    # 将纬度和经度从度转换为弧度
    s1 = s1 * math.pi / 180
    s2 = s2 * math.pi / 180

    # 调整s1和s2的形状以便于广播
    s1_expanded = s1.unsqueeze(2)  # [batch_size, seq_len, 1, 2]
    s2_expanded = s2.unsqueeze(1)  # [batch_size, 1, seq_len, 2]

    # 计算差异
    dlat = s2_expanded[..., 0] - s1_expanded[..., 0]
    dlon = s2_expanded[..., 1] - s1_expanded[..., 1]

    # Haversine公式
    a = torch.sin(dlat / 2)**2 + torch.cos(s1_expanded[..., 0]) * torch.cos(s2_expanded[..., 0]) * torch.sin(dlon / 2)**2
    c = 2 * torch.asin(torch.sqrt(a))

    # 地球平均半径，单位千米
    r = 6371

    # 计算最终距离
    return c * r  # [batch_size, seq_len, seq_len]

class Rnn(Enum):
    """ The available RNN units """

    RNN = 0
    GRU = 1
    LSTM = 2

    @staticmethod
    def from_string(name):
        if name == 'rnn':
            return Rnn.RNN
        if name == 'gru':
            return Rnn.GRU
        if name == 'lstm':
            return Rnn.LSTM
        raise ValueError('{} not supported in --rnn'.format(name))


class RnnFactory():
    """ Creates the desired RNN unit. """

    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)

    def __str__(self):
        if self.rnn_type == Rnn.RNN:
            return 'Use pytorch RNN implementation.'
        if self.rnn_type == Rnn.GRU:
            return 'Use pytorch GRU implementation.'
        if self.rnn_type == Rnn.LSTM:
            return 'Use pytorch LSTM implementation.'

    def is_lstm(self):
        return self.rnn_type in [Rnn.LSTM]

    def create(self, hidden_size):
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size) 
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size)
    


class AttentionLayer(nn.Module):
    def __init__(self, user_dim, item_dim):
        super(AttentionLayer, self).__init__()
        self.attention_fc = nn.Sequential(
            nn.Linear(user_dim + item_dim, 32),  # Suppose the hidden layer size is 128
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )
        for layer in self.attention_fc:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)
    
    def forward(self, user_embeddings, item_embeddings, edge_index):
        # Extract edges' user and item nodes' embeddings
        user_indices = edge_index[0]
        item_indices = edge_index[1]
        user_feats = user_embeddings[user_indices]
        item_feats = item_embeddings[item_indices]

        # Concatenate user and item embeddings to compute edge weights
        edge_feats = torch.cat([user_feats, item_feats], dim=1)
        edge_weights = torch.sigmoid(self.attention_fc(edge_feats)).squeeze()

        return edge_weights

class DenoisingLayer(nn.Module):
    def __init__(self):
        super(DenoisingLayer, self).__init__()

    def forward(self, edge_weights, edge_index, threshold=0.8):
        # Apply a threshold to filter edges
        mask = edge_weights > threshold
        if mask.sum() == 0:
            # 如果所有的权重都低于阈值，保留一个最大权重的边避免孤立节点
            mask[edge_weights.argmax()] = True
        denoised_edge_index = edge_index[:, mask]
        denoised_edge_weights = edge_weights[mask]

        return denoised_edge_index, denoised_edge_weights

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x

# Define the complete model
class DenoisingGCNNet(nn.Module):
    def __init__(self, user_dim, item_dim, out_channels):
        super(DenoisingGCNNet, self).__init__()
        self.attention_layer = AttentionLayer(user_dim, item_dim)
        self.denoising_layer = DenoisingLayer()
        self.gcn_layer = GCNLayer(user_dim, out_channels)  

    def forward(self, user_embeddings, item_embeddings, edge_index):
        # Compute attention weights for each edge
        edge_weights = self.attention_layer(user_embeddings, item_embeddings, edge_index)
        
        # Filter edges to create a denoised graph
        denoised_edge_index, denoised_edge_weights = self.denoising_layer(edge_weights, edge_index)
        
        # Combine user and item embeddings for GCN input
        gcn_input = torch.cat([user_embeddings, item_embeddings], dim=0)
        
        # Apply GCN on the denoised graph
        gcn_output = self.gcn_layer(gcn_input, denoised_edge_index)
        
        return gcn_output, denoised_edge_index, denoised_edge_weights

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class DenoisingNet(nn.Module):
    def __init__(self, gcnLayers, num_features, num_users, num_items, hidden_dim):
        super(DenoisingNet, self).__init__()

        self.gcn_layers = nn.ModuleList([GraphConvolution(num_features, hidden_dim) for _ in range(gcnLayers)])

        self.nblayers_0 = nn.Sequential(nn.Linear(num_features, hidden_dim), nn.LeakyReLU(inplace=True))
        self.nblayers_1 = nn.Sequential(nn.Linear(num_features, hidden_dim), nn.LeakyReLU(inplace=True))

        self.selflayers_0 = nn.Sequential(nn.Linear(num_features, hidden_dim), nn.LeakyReLU(inplace=True))
        self.selflayers_1 = nn.Sequential(nn.Linear(num_features, hidden_dim), nn.LeakyReLU(inplace=True))

        self.attentions_0 = nn.Sequential(nn.Linear(2 * hidden_dim, 1))
        self.attentions_1 = nn.Sequential(nn.Linear(2 * hidden_dim, 1))

        self.num_users = num_users
        self.num_items = num_items

    def get_attention(self, input1, input2, layer=0):
        if layer == 0:
            nb_layer = self.nblayers_0
            selflayer = self.selflayers_0
        elif layer == 1:
            nb_layer = self.nblayers_1
            selflayer = self.selflayers_1

        input1 = nb_layer(input1)
        input2 = selflayer(input2)

        input10 = torch.concat([input1, input2], axis=1)

        if layer == 0:
            weight10 = self.attentions_0(input10)
        elif layer == 1:
            weight10 = self.attentions_1(input10)

        return weight10

    def generate(self, x, adj_mat, layer=0):
        self.row, self.col = adj_mat._indices()

        f1_features = x[self.row, :]
        f2_features = x[self.col, :]

        weight = self.get_attention(f1_features, f2_features, layer)
        mask = torch.sigmoid(weight)
        mask = torch.squeeze(mask)

        adj = torch.sparse.FloatTensor(adj_mat._indices(), mask, adj_mat.shape)
        return adj

    def forward(self, x, adj_mat):
        layer_index = 0
        embedsLst = [x] 

        for layer in self.gcn_layers:
            adj = self.generate(x, adj_mat, layer=layer_index)
            x = F.relu(layer(x, adj))
            embedsLst.append(x)
            layer_index += 1

        return sum(embedsLst)

        
class Flashback(nn.Module):
    """ Flashback RNN: Applies weighted average using spatial and tempoarl data in combination
    of user embeddings to the output of a generic RNN unit (RNN, GRU, LSTM).
    """

    def __init__(self, input_size, user_count, hidden_size, f_t, f_s, rnn_factory, lambda_loc, lambda_user, use_weight,
                 graph, spatial_graph, friend_graph, use_graph_user, use_spatial_graph, interact_graph, setting):
        super().__init__()
        self.input_size = input_size  # POI个数
        self.user_count = user_count
        self.hidden_size = hidden_size
        self.f_t = f_t  # function for computing temporal weight
        self.f_s = f_s  # function for computing spatial weight

        self.lambda_loc = lambda_loc
        self.lambda_user = lambda_user
        self.use_weight = use_weight
        self.use_graph_user = use_graph_user
        self.use_spatial_graph = use_spatial_graph

        self.I = identity(graph.shape[0], format='coo')
        self.graph = sparse_matrix_to_tensor(
            calculate_random_walk_matrix((graph * self.lambda_loc + self.I).astype(np.float32)))

        self.spatial_graph = spatial_graph
        if interact_graph is not None:
            self.interact_graph = sparse_matrix_to_tensor(calculate_random_walk_matrix(
                interact_graph))  # (M, N)
        else:
            self.interact_graph = None

        self.encoder = nn.Embedding(
            input_size, hidden_size)  # location embedding
        # self.time_encoder = nn.Embedding(24 * 7, hidden_size)  # time embedding
        self.user_encoder = nn.Embedding(
            user_count, hidden_size)  # user embedding
        self.rnn = rnn_factory.create(hidden_size) 
        
        self.setting = setting



        self.seq_model = EncoderLayer(
                                setting.hidden_dim+6,
                                setting.transformer_nhid,
                                setting.transformer_dropout,
                                setting.attention_dropout_rate,
                                setting.transformer_nhead)
        self.time_embed_model = Time2Vec('sin', setting.batch_size, setting.sequence_length, out_dim=6)
        self.embed_fuse_model = FuseEmbeddings(hidden_size, 6)

        self.decoder = nn.Linear(setting.hidden_dim+6, setting.hidden_dim)
        self.fc = nn.Linear(2*hidden_size, input_size)

        self.time_decoder = nn.Linear(setting.hidden_dim+6, 1)

        self.denoise = DenoisingGCNNet(self.hidden_size, self.hidden_size, self.hidden_size)


    def forward(self, x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_user, f, y_f, dataset):
        seq_len, user_len = x.size()
        x_emb = self.encoder(x)


        # 是否用GCN来更新user embedding
        if self.use_graph_user:
            # I_f = identity(self.friend_graph.shape[0], format='coo')
            # friend_graph = (self.friend_graph * self.lambda_user + I_f).astype(np.float32)
            # friend_graph = calculate_random_walk_matrix(friend_graph)
            # friend_graph = sparse_matrix_to_tensor(friend_graph).to(x.device)
            friend_graph = self.friend_graph.to(x.device)
            # AX
            user_emb = self.user_encoder(torch.LongTensor(
                list(range(self.user_count))).to(x.device))
            user_encoder_weight = torch.sparse.mm(friend_graph, user_emb).to(
                x.device)  # (user_count, hidden_size)

            if self.use_weight:
                user_encoder_weight = self.user_gconv_weight(
                    user_encoder_weight)
            p_u = torch.index_select(
                user_encoder_weight, 0, active_user.squeeze())
        else:
            p_u = self.user_encoder(active_user)  # (1, user_len, hidden_size)
            # (user_len, hidden_size)
            p_u = p_u.view(user_len, self.hidden_size)

        p_u = self.user_encoder(active_user)  # (1, user_len, hidden_size)
        p_u = p_u.view(user_len, self.hidden_size)
        # AX,即GCN
        graph = self.graph.to(x.device)
        loc_emb = self.encoder(torch.LongTensor(
            list(range(self.input_size))).to(x.device))
        # loc_emb = poi_embeddings[1:]
        encoder_weight = torch.sparse.mm(graph, loc_emb).to(
            x.device)  # (input_size, hidden_size)
        
        if self.use_spatial_graph:
            spatial_graph = (self.spatial_graph *
                             self.lambda_loc + self.I).astype(np.float32)
            spatial_graph = calculate_random_walk_matrix(spatial_graph)
            spatial_graph = sparse_matrix_to_tensor(
                spatial_graph).to(x.device)  # sparse tensor gpu
            encoder_weight += torch.sparse.mm(spatial_graph,
                                              loc_emb).to(x.device)
            encoder_weight /= 2  # 求均值
       
        new_x_emb = []
        for i in range(seq_len):
            # (user_len, hidden_size)
            temp_x = torch.index_select(encoder_weight, 0, x[i])
            new_x_emb.append(temp_x)

        x_emb = torch.stack(new_x_emb, dim=0)  

        # user-poi
        loc_emb = self.encoder(torch.LongTensor(
            list(range(self.input_size))).to(x.device))
        encoder_weight = loc_emb
        interact_graph = self.interact_graph.to(x.device)
        encoder_weight_user = torch.sparse.mm(
            interact_graph, encoder_weight).to(x.device)
        
        user_emb = self.encoder(torch.LongTensor(
            list(range(self.interact_graph.size(0)))).to(x.device))
        encoder_weight = user_emb
        encoder_weight_poi = torch.sparse.mm(
             interact_graph.t(), encoder_weight).to(x.device)
        

        edge_index = self.interact_graph.coalesce().indices().to(x.device)

        gcn_output, denoised_edge_index, denoised_edge_weights = self.denoise(encoder_weight_user, encoder_weight_poi, edge_index)

        # encoder_weight_user= gcn_output[:self.interact_graph.size(0)]
        encoder_weight_poi = gcn_output[self.interact_graph.size(0):]


        new_x_emb = []
        for i in range(seq_len):
            temp_x = torch.index_select(encoder_weight_poi, 0, x[i])
            new_x_emb.append(temp_x)

        x_emb_new = torch.stack(new_x_emb, dim=0) 

        x_emb = (x_emb +x_emb_new)/2

        user_preference = torch.index_select(
            encoder_weight_user, 0, active_user.squeeze()).unsqueeze(0)
        # print(user_preference.size())
        user_loc_similarity = torch.exp(
            -(torch.norm(user_preference - x_emb, p=2, dim=-1))).to(x.device)
        user_loc_similarity = user_loc_similarity.permute(1, 0)

        # out, h = self.rnn(x_emb, h)  # (seq_len, user_len, hidden_size)

        # src_mask = self.seq_model.generate_square_subsequent_mask(self.setting.batch_size).to(x.device)

        t_emb = self.time_embed_model(t_slot.transpose(0,1)/168).to(x.device)
        x_emb = self.embed_fuse_model(x_emb.transpose(0,1), t_emb).to(x.device)
        # dist_attn = (haversines(s.transpose(0,1), s.transpose(0,1))).unsqueeze(-1).repeat_interleave(2, dim=-1)
        # dist_attn_fs = self.f_s(dist_attn, user_len).transpose(1, -1)
        # dist_attn_fs = 1/(1+dist_attn).transpose(1, -1)
        out = self.seq_model(x_emb).to(x.device)
        # out = self.seq_model(x_emb).to(x.device)
        out_time = self.time_decoder(out).to(x.device).transpose(0,1)

        # out = self.decoder_poi(out).to(x.device).transpose(0,1)
        out = self.decoder(out).to(x.device).transpose(0,1)




        out_w = torch.zeros(seq_len, user_len,
                            self.hidden_size, device=x.device)
        
        for i in range(seq_len):
            sum_w = torch.zeros(user_len, 1, device=x.device)  # (200, 1)
            for j in range(i + 1):
                dist_s = haversine(s[i], s[j])
                # a_j = self.f_t(dist_t, user_len)  # (user_len, )
                b_j = self.f_s(dist_s, user_len)
                b_j = b_j.unsqueeze(1)
                w_j = b_j + 1e-10 
                w_j = w_j * user_loc_similarity[:, j].unsqueeze(1)  # (user_len, 1)
                sum_w += w_j
                out_w[i] += w_j * out[j]  # (user_len, hidden_size)
            out_w[i] /= sum_w
        

            
        out_pu = torch.zeros(seq_len, user_len, 2 *
                             self.hidden_size, device=x.device)
        for i in range(seq_len):
            # (user_len, hidden_size * 2)
            out_pu[i] = torch.cat([out_w[i], p_u], dim=1)

        # 计算特征向量和对应类别中心向量的余弦相似度
        cosine_similarity = F.linear(F.normalize(out_pu), F.normalize(self.fc.weight))

        y_linear = self.fc(out_pu)  # (seq_len, user_len, loc_count)

        

        return y_linear, h, cosine_similarity, out_time


'''
~~~ h_0 strategies ~~~
Initialize RNNs hidden states
'''


def create_h0_strategy(hidden_size, is_lstm):
    if is_lstm:
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))
    else:
        return FixNoiseStrategy(hidden_size)


class H0Strategy():

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def on_init(self, user_len, device):
        pass

    def on_reset(self, user):
        pass

    def on_reset_test(self, user, device):
        return self.on_reset(user)


class FixNoiseStrategy(H0Strategy):
    """ use fixed normal noise as initialization """

    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1 / self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu

    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        # (1, 200, 10)
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)

    def on_reset(self, user):
        return self.h0


class LstmStrategy(H0Strategy):
    """ creates h0 and c0 using the inner strategy """

    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy

    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return h, c

    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return h, c


"""
class BipartiteGraphDenoisingLayer(nn.Module):
    # 专门为用户-POI交互图设计的去噪层
    def __init__(self, num_users, num_pois, num_features, dropout_rate=0.5):
        super(BipartiteGraphDenoisingLayer, self).__init__()
        # 注意力机制的实现，针对用户-POI边
        self.user_poi_attention = nn.Parameter(torch.Tensor(num_users, num_pois))
        self.dropout_rate = dropout_rate
        self.reset_parameters()
        self.num_users=num_users
        self.num_pois=num_pois

    def reset_parameters(self):
        # 参数初始化
        nn.init.xavier_uniform_(self.user_poi_attention)

    def forward(self, user_features, poi_features, user_poi_edge_index):
        # 去噪逻辑
        user_poi_edge_weights = torch.sigmoid(self.user_poi_attention)
        user_poi_edge_weights = F.dropout(user_poi_edge_weights, p=self.dropout_rate)
        
        # 构建去噪后的用户-POI边权重矩阵
        denoised_user_poi_adj = torch.sparse.FloatTensor(user_poi_edge_index, user_poi_edge_weights, torch.Size([self.num_users, self.num_pois]))
        return denoised_user_poi_adj
bipartite_graph_denoising = BipartiteGraphDenoisingLayer(num_users=self.interact_graph.size(0), num_pois=self.input_size, num_features=self.hidden_size)
bipartite_graph_denoising(encoder_weight_user, encoder_weight, self.interact_graph.coalesce().indices())
"""