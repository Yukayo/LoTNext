import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, embed_size, nhead, nhid, nlayers, max_len, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, max_len, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        # self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # def init_weights(self):
    #     initrange = 0.1
    #     self.decoder_poi.bias.data.zero_()
    #     self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        # src = self.encoder(src)
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        return x
        # out_poi = self.decoder_poi(x)
        # return out_poi


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None, mask=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        if mask is not None:
            mask = mask.unsqueeze(1)
            x = x.masked_fill(mask, 0)

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads
    ):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm1 = nn.LayerNorm(hidden_size)
        self.ffn_norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, mask=None):
        # y = self.self_attention_norm(x)
        # y = self.self_attention(y, y, y, attn_bias, mask=mask)
        y = self.self_attention(x, x, x, attn_bias, mask=mask)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm1(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        x = self.ffn_norm2(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_weights = F.softmax(Q @ K.transpose(-2, -1) / math.sqrt(K.size(-1)), dim=-1)
        return attention_weights @ V
    
class Expert(nn.Module):
    def __init__(self, num_pois, embedding_dim, hidden_dim):
        super(Expert, self).__init__()
        self.poi_embedding = nn.Embedding(num_pois, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, poi_sequences):
        embedded_sequences = self.poi_embedding(poi_sequences)
        lstm_out, (h_n, c_n) = self.lstm(embedded_sequences)
        mlp_output = self.mlp(h_n[-1])
        return mlp_output

class EnsembleModel(nn.Module):
    def __init__(self, num_pois, embedding_dim, hidden_dim, num_experts):
        super(EnsembleModel, self).__init__()
        self.experts = nn.ModuleList(
            [Expert(num_pois, embedding_dim, hidden_dim) for _ in range(num_experts)]
        )
        self.attention = SelfAttention(embedding_dim)
        self.num_pois = num_pois

    def forward(self, expert_batches, expert_poi_indices):
        # 存储每个专家的输出
        expert_outputs = torch.zeros((len(expert_batches), self.num_pois, self.attention.embedding_dim), device=expert_batches[0].device)
        # 存储每个POI的出现次数，用于之后的平均操作
        poi_counts = torch.zeros(self.num_pois, device=expert_batches[0].device)
        
        for expert_idx, batches in enumerate(expert_batches):
            for batch, poi_indices in zip(batches, expert_poi_indices[expert_idx]):
                # 获取当前专家模型的输出
                expert_output = self.experts[expert_idx](batch)
                # 根据POI索引累加输出
                for idx, poi_idx in enumerate(poi_indices):
                    expert_outputs[expert_idx, poi_idx] += expert_output[idx]
                    poi_counts[poi_idx] += 1
        
        # 将专家输出按POI出现次数平均
        expert_outputs /= poi_counts.unsqueeze(1).unsqueeze(0)

        # 使用自注意力融合所有专家的输出
        aggregated_outputs = self.attention(expert_outputs.view(-1, self.attention.embedding_dim))
        
        return aggregated_outputs

def t2v(tau, f, out_features, w, b, w0, b0):
    # tau [batch_size, seq_len], w [seq_len, out_features - 1], b [out_features - 1]
    # w0 [seq_len, 1], b0 [1]
    v1 = f(tau.unsqueeze(-1) * w + b)  # [batch_size, seq_len, out_features - 1]
    v2 = tau.unsqueeze(-1) * w0 + b0   # [batch_size, seq_len, 1]
    return torch.cat([v1, v2], -1)  # [batch_size, seq_len, out_features]

class SineActivation(nn.Module):
    def __init__(self, batch_size, seq_len, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(batch_size, seq_len, 1))
        self.b0 = nn.Parameter(torch.randn(batch_size, seq_len, 1))
        self.w = nn.Parameter(torch.randn(batch_size, seq_len, out_features - 1))
        self.b = nn.Parameter(torch.randn(batch_size, seq_len, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        # tau [batch_size, seq_len]
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class SineActivation1(nn.Module):
    def __init__(self, seq_len, out_features):
        super(SineActivation1, self).__init__()
        self.out_features = out_features
        self.l1 = nn.Linear(1, out_features-1, bias=True)
        self.l2 = nn.Linear(1, 1, bias=True)
        self.f = torch.sin

    def forward(self, tau):
        v1 = self.l1(tau.unsqueeze(-1))
        v1 = self.f(v1)
        v2 = self.l2(tau.unsqueeze(-1))
        return torch.cat([v1, v2], -1)

class Time2Vec(nn.Module):
    def __init__(self, activation, batch_size, seq_len, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation1(seq_len, out_dim)
            # self.l1 = SineActivation(batch_size, seq_len, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(seq_len, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x
    
class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 2))
        x = self.leaky_relu(x)
        return x

class FuseEmbeddings1(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings1, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, user_embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 0))
        x = self.leaky_relu(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Linear(user_embed_dim, poi_embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed):
        x = self.decoder(user_embed)
        x = self.leaky_relu(x)
        return x