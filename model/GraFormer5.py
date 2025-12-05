from __future__ import absolute_import

import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as sp
import copy, math
import torch.nn.functional as F
from torch.nn.functional import layer_norm
from torch.nn.parameter import Parameter
from model.blocks.ChebConv import ChebConv, _ResChebGC
from model.blocks.refine import refine
from common.utils import *

edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                      [0, 4], [4, 5], [5, 6],
                      [0, 7], [7, 8], [8, 9], [9, 10],
                      [8, 11], [11, 12], [12, 13],
                      [8, 14], [14, 15], [15, 16]], dtype=torch.long)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


gan_edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4],
                          [0, 5], [5, 6], [6, 7], [7, 8],
                          [0, 9], [9, 10], [10, 11], [11, 12],
                          [0, 13], [13, 14], [14, 15], [15, 16],
                          [0, 17], [17, 18], [18, 19], [19, 20]], dtype=torch.long)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # features=layer.size=512
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(size, 3 * size, bias=True)
        )

    def forward(self, x, sublayer, c):
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=2)
        return x + gate * self.dropout(sublayer(modulate(self.norm(x), shift, scale)))


class GraAttenLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(GraAttenLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, pre_3d, mask, mask2, c):
        x = self.sublayer[0](x, lambda x: self.self_attn(pre_3d, x, x, mask, mask2), c)
        return self.sublayer[1](x, self.feed_forward, c)


import random


def attention(Q, K, V, mask=None, mask2=None, dropout=None):
    # Query=Key=Value: [batch_size, 8, max_len, 64]
    d_k = Q.size(-1)
    # Q * K.T = [batch_size, 8, max_len, 64] * [batch_size, 8, 64, max_len]
    # scores: [batch_size, 8, max_len, max_len]
    # scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # padding mask
    if mask2 is not None:
        # scores = scores.masked_fill(mask == 0, -1e9)
        B, H, J, _ = Q.shape
        body_parts = {
            'right_leg': [4, 5, 6],
            'left_leg': [1, 2, 3],
            'torso': [0, 7, 8, 9, 10],
            'left_arm': [14, 15, 16],
            'right_arm': [11, 12, 13]
        }
        joint_hierarchy = {
            0: 1,
            1: 2, 4: 2, 7: 2,
            2: 3, 5: 3, 8: 3,
            3: 4, 6: 4, 9: 4, 11: 4, 14: 4,
            10: 5, 12: 5, 15: 5,
            13: 6, 16: 6
        }
        spatial_mask = torch.zeros(B, J, dtype=torch.bool)
        for b in range(B):
            if random.random() < 0.1:
                part = random.choice(list(body_parts.keys()))
                idx = body_parts[part]
                spatial_mask[b, idx] = True
            if random.random() < 0.2:
                hierarchy_level = random.choice(list(set(joint_hierarchy.values())))
                idx = [j for j, h in joint_hierarchy.items() if h == hierarchy_level]
                spatial_mask[b, idx] = True
        mask_expanded_Q = spatial_mask.unsqueeze(1).unsqueeze(-1).to(Q.device)  # [B, 1, J, 1]
        mask_token = nn.Parameter(torch.randn(1, 1, 1, d_k, device=Q.device))
        Q = torch.where(mask_expanded_Q, mask_token.expand_as(Q), Q)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, V), p_attn


# yes-mus
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        # »ù´¡¹Ç¼ÜÈ¨ÖØ
        self.body_parts_weight = nn.Parameter(torch.tensor(1.0))  # ÉíÌå²¿Î»ÑÚÂëÈ¨ÖØ
        self.direct_connect_weight = nn.Parameter(torch.tensor(2.0))  # Ö±½ÓÁ¬½áÔöÇ¿È¨ÖØ
        self.symmetric_weight = nn.Parameter(torch.tensor(1.5))  # ¶Ô³Æ¹Ø½ÚÈ¨ÖØ
        self.self_connect_weight = nn.Parameter(torch.tensor(3.0))  # ×ÔÁ¬½ÓÈ¨ÖØ

        # ¶¯Ì¬¹Ç¼ÜÊÊÅäÆ÷ÍøÂç
        self.skeleton_adapter = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(),
        )

        # ¹Ø½Ú¹ØÏµ±àÂëÆ÷ - ÓÃÓÚÔ¤²â¹Ø½Ú¼ä¹ØÏµµÄÇ¿¶È
        self.joint_relation_predictor = nn.Linear(d_model // 4, 17)

        # ¶¯Ì¬µ÷ÕûÏµÊýµÄËõ·ÅÒò×Ó
        self.dynamic_scale = nn.Parameter(torch.tensor(0.5))  # ³õÊ¼Öµ½ÏÐ¡£¬ÈÃÄ£ÐÍ´Ó»ù´¡ÑÚÂë¿ªÊ¼Ñ§Ï°

    def _create_skeleton_mask(self):
        """´´½¨»ù´¡¹Ç¼ÜÑÚÂë"""
        # »ù´¡ÑÚÂë - È«1¾ØÕó
        mask = torch.ones(17, 17).to('cuda')

        # ¶¨ÒåÖ±½ÓÁ¬½ÓµÄ¹Ø½Ú¶Ô(¸¸×Ó¹ØÏµ)
        direct_pairs = [
            (0, 1), (1, 2), (2, 3),  # ÓÒÍÈ
            (0, 4), (4, 5), (5, 6),  # ×óÍÈ
            (0, 7), (7, 8), (8, 9), (9, 10),  # ¼¹Öù
            (8, 11), (11, 12), (12, 13),  # ÓÒ±Û
            (8, 14), (14, 15), (15, 16)  # ×ó±Û
        ]

        # ÔöÇ¿Ö±½ÓÁ¬½ÓµÄ¹Ø½Ú
        for i, j in direct_pairs:
            mask[i, j] += self.direct_connect_weight

        # ÔöÇ¿¶Ô³Æ¹Ø½Ú
        symmetric_pairs = [(1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16)]
        for i, j in symmetric_pairs:
            mask[i, j] += self.symmetric_weight
            mask[j, i] += self.symmetric_weight

        # ÔöÇ¿×ÔÉíÁ¬½Ó
        for i in range(17):
            mask[i, i] += self.self_connect_weight

        return mask

    def _create_dynamic_skeleton_mask(self, query_features):
        """´´½¨¶¯Ì¬¹Ç¼ÜÑÚÂë"""
        batch_size, n_joints, dim = query_features.shape

        # 1. ÌáÈ¡Ã¿¸ö¹Ø½ÚµÄÌØÕ÷±íÊ¾
        joint_features = self.skeleton_adapter(query_features)  # [batch, joints, dim//4]

        # 2. ÎªÃ¿¸ö¹Ø½ÚÔ¤²âÓëÆäËû¹Ø½ÚµÄ¹ØÏµÇ¿¶È
        relation_logits = []
        for i in range(n_joints):
            # Ê¹ÓÃµ±Ç°¹Ø½ÚÌØÕ÷Ô¤²âÓëÆäËû¹Ø½ÚµÄ¹ØÏµ
            joint_relation = self.joint_relation_predictor(joint_features[:, i])  # [batch, joints]
            relation_logits.append(joint_relation)

        # ½«¹ØÏµºÏ²¢ÎªÍêÕûµÄ¹ØÏµ¾ØÕó
        relation_matrix = torch.stack(relation_logits, dim=1)  # [batch, joints, joints]

        # Ê¹ÓÃsigmoid½«¹ØÏµÇ¿¶ÈÏÞÖÆÔÚ0-1Ö®¼ä£¬È»ºóÓ¦ÓÃËõ·ÅÒò×Ó
        dynamic_adjustments = torch.sigmoid(relation_matrix) * self.dynamic_scale

        # »ñÈ¡»ù´¡¹Ç¼ÜÑÚÂë²¢À©Õ¹µ½Åú´ÎÎ¬¶È
        base_mask = self._create_skeleton_mask()
        base_mask_expanded = base_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # ½«¶¯Ì¬µ÷ÕûÓ¦ÓÃµ½»ù´¡ÑÚÂëÉÏ
        dynamic_mask = base_mask_expanded + dynamic_adjustments

        return dynamic_mask

    def forward(self, query, key, value, mask=None, mask2=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) ÏßÐÔ±ä»»²¢·Ö¸îÍ·
        Q, K, V = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (query, key, value))]

        # 2) ¼ÆËã×¢ÒâÁ¦·ÖÊý
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3) Éú³É¶¯Ì¬¹Ç¼ÜÑÚÂë
        # Ê×ÏÈÆ½¾ù¶à¸ö×¢ÒâÁ¦Í·µÄÌØÕ÷£¬ÓÃÓÚ¹Ç¼ÜÊÊÅä
        avg_query_features = query  # Ê¹ÓÃÔ­Ê¼²éÑ¯ÌØÕ÷£¬ÐÎ×´Îª[batch, joints, dim]
        dynamic_skeleton_mask = self._create_dynamic_skeleton_mask(avg_query_features)

        # À©Õ¹ÑÚÂëÒÔÊÊÓ¦¶àÍ·×¢ÒâÁ¦¸ñÊ½ [batch, heads, joints, joints]
        dynamic_mask_expanded = dynamic_skeleton_mask.unsqueeze(1).expand(-1, self.h, -1, -1)

        # 4) Ó¦ÓÃ¶¯Ì¬¹Ç¼ÜÑÚÂëÖ¸µ¼×¢ÒâÁ¦
        guided_scores = scores * dynamic_mask_expanded

        # 5) Ó¦ÓÃpadding mask
        if mask is not None:
            guided_scores = guided_scores.masked_fill(mask == 0, -1e9)

        # 6) ¼ÆËã×¢ÒâÁ¦È¨ÖØ
        p_attn = F.softmax(guided_scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        # 7) Ó¦ÓÃ×¢ÒâÁ¦È¨ÖØµ½values
        x = torch.matmul(p_attn, V)

        # 8) Æ´½Ó½á¹û
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)

        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True, True, True]]])


class LAM_Gconv(nn.Module):

    def __init__(self, in_features, out_features, activation=nn.ReLU(inplace=True)):
        super(LAM_Gconv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation

    def laplacian(self, A_hat):
        D_hat = (torch.sum(A_hat, 0) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L

    def laplacian_batch(self, A_hat):
        batch, N = A_hat.shape[:2]
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, X, A):
        batch = X.size(0)
        A_hat = A.unsqueeze(0).repeat(batch, 1, 1)
        X = self.fc(torch.bmm(self.laplacian_batch(A_hat), X))
        if self.activation is not None:
            X = self.activation(X)
        return X


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class GraphNet(nn.Module):

    def __init__(self, in_features=2, out_features=2, n_pts=21):
        super(GraphNet, self).__init__()

        self.A_hat = Parameter(torch.eye(n_pts).float(), requires_grad=True)  # Éú³ÉÒ»¸ö¶Ô½Ç¾ØÕó
        self.gconv1 = LAM_Gconv(in_features, in_features * 2)
        self.gconv2 = LAM_Gconv(in_features * 2, out_features, activation=None)

    def forward(self, X):
        X_0 = self.gconv1(X, self.A_hat)
        X_1 = self.gconv2(X_0, self.A_hat)
        return X_1


class FinalLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.gconv = ChebConv(in_c=dim, out_c=3, K=2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True)
        )

    def forward(self, x, adj, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm(x), shift, scale)
        x = self.gconv(x, adj)
        return x


from model.blocks.gcn_conv import Gcn_block
from model.blocks.graph_frames import Graph
from timm.models.layers import DropPath
from model.blocks.drop import DropPath as DropPath2  # ÊÇÒÔÏÂ
from functools import partial

from common.opt import opts

opt = opts().parse()
boneindextemp = opt.boneindex_h36m.split(',')
boneindex = []
for i in range(0, len(boneindextemp), 2):
    boneindex.append([int(boneindextemp[i]), int(boneindextemp[i + 1])])


class gcn1(nn.Module):
    def __init__(self, dim, h_dim, drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        self.graph = Graph('hm36_gt', 'spatial', pad=0)
        self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False).cuda()
        kernel_size = self.A.size(0)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gcn1 = Gcn_block(dim, h_dim, kernel_size, residual=False)
        self.norm_gcn1 = norm_layer(dim)
        self.gcn2 = Gcn_block(h_dim, dim, kernel_size, residual=False)
        self.norm_gcn2 = norm_layer(dim)

    def forward(self, x):
        res = x
        x, A = self.gcn1(self.norm_gcn1(x), self.A)
        x, A = self.gcn2(x, self.A)
        x = res + self.drop_path(self.norm_gcn2(x))
        return x


class PartBasedGraphConv(nn.Module):
    def __init__(self, input_dim, part_dim=None):
        super(PartBasedGraphConv, self).__init__()
        part_dim = input_dim if part_dim is None else part_dim

        # ¶¨ÒåÉíÌå5¸öÖ÷Òª²¿Î»¼°Æä¹Ø½ÚË÷Òý
        self.body_parts = {
            'right_leg': [1, 2, 3],  # ÓÒÍÈ
            'left_leg': [4, 5, 6],  # ×óÍÈ
            'torso': [0, 7, 8, 9, 10],  # Çû¸É
            'right_arm': [11, 12, 13],  # ÓÒ±Û
            'left_arm': [14, 15, 16]  # ×ó±Û
        }

        # ¶¨Òå²¿Î»Ö®¼äµÄ±ßÁ¬½Ó - »ùÓÚÈËÌå½âÆÊ½á¹¹
        part_edges = [
            [0, 2],  # right_leg - torso
            [1, 2],  # left_leg - torso
            [2, 3],  # torso - right_arm
            [2, 4]  # torso - left_arm
        ]

        # Ê¹ÓÃadj_mx_from_edges´´½¨²¿Î»ÁÚ½Ó¾ØÕó
        self.register_buffer('part_adj', adj_mx_from_edges(num_pts=5, edges=part_edges, sparse=False))

        # ²¿Î»¼¶ChebConv
        self.part_conv = ChebConv(in_c=input_dim, out_c=part_dim, K=2)

        # ÎªÃ¿¸ö²¿Î»´´½¨×ÔÊÊÓ¦³Ø»¯²ã
        self.pools = nn.ModuleDict({
            part_name: nn.AdaptiveAvgPool1d(1)
            for part_name in self.body_parts.keys()
        })

        # Ñ§Ï°È¨ÖØÆ½ºâÔ­Ê¼ÌØÕ÷ºÍ²¿Î»ÌØÕ÷
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        batch_size, n_joints, feat_dim = x.shape
        device = x.device

        # 1. Ê¹ÓÃAvgPool½«¹Ø½ÚÌØÕ÷¾ÛºÏµ½²¿Î»
        part_features = []

        for part_name, joint_indices in self.body_parts.items():
            # Ñ¡Ôñ¸Ã²¿Î»µÄ¹Ø½Ú
            joints_feats = x[:, joint_indices, :]  # [batch, part_size, feat_dim]

            # µ÷ÕûÎ¬¶ÈÒÔÊÊÓ¦³Ø»¯²Ù×÷
            joints_feats = joints_feats.permute(0, 2, 1)  # [batch, feat_dim, part_size]

            # Ó¦ÓÃ³Ø»¯
            pool = self.pools[part_name]
            pooled_feat = pool(joints_feats)  # [batch, feat_dim, 1]
            pooled_feat = pooled_feat.squeeze(-1)  # [batch, feat_dim]

            part_features.append(pooled_feat)

        # ½«²¿Î»ÌØÕ÷¶Ñµþ [batch, 5, feat_dim]
        part_features = torch.stack(part_features, dim=1)

        # 2. ¶Ô²¿Î»ÌØÕ÷Ö´ÐÐÍ¼¾í»ý
        part_out = self.part_conv(part_features, self.part_adj.to(device))

        # 3. ½«²¿Î»ÌØÕ÷·ÖÅä»Ø¸÷¸ö¹Ø½Ú
        joint_features = torch.zeros_like(x)

        for part_idx, (part_name, joint_indices) in enumerate(self.body_parts.items()):
            # »ñÈ¡µ±Ç°²¿Î»ÌØÕ÷
            part_feat = part_out[:, part_idx]  # [batch, feat_dim]

            # ½«ÌØÕ÷·ÖÅä¸ø¸Ã²¿Î»µÄËùÓÐ¹Ø½Ú
            for joint_idx in joint_indices:
                joint_features[:, joint_idx] = part_feat

        # 4. ÈÚºÏÔ­Ê¼¹Ø½ÚÌØÕ÷ºÍ²¿Î»ÌØÕ÷
        output = joint_features

        return output


class GraFormer(nn.Module):
    def __init__(self, hid_dim=128, coords_dim=(5, 3), num_layers=4, n_head=4, dropout=0.1, n_pts=17, is_train=True,
                 mlp_hidden_dim=96 * 4, drop_rate=0.1, ):
        super(GraFormer, self).__init__()
        self.n_layers = num_layers
        self.is_train = is_train

        self.adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False)

        self.src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True,
                                        True, True, True, True, True, True, True]]]).cuda()

        _gconv_input = ChebConv(in_c=2, out_c=hid_dim, K=2)

        _gconv_cond = ChebConv(in_c=2, out_c=hid_dim, K=2)
        _gconv_layers = []
        # _gconv_layers1 = []
        _attention_layer = []

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hid_dim),
            nn.Linear(hid_dim, hid_dim * 2),
            # nn.GELU(),
            nn.SiLU(),
            nn.Linear(hid_dim * 2, hid_dim),
        )

        dim_model = hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, dim_model)
        gcn = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=n_pts)

        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC(adj=self.adj, input_dim=hid_dim, output_dim=hid_dim,
                                            hid_dim=hid_dim, p_dropout=0.1))
            # _gconv_layers1.append(gcn1(dim=hid_dim, h_dim=2*hid_dim, drop_path=dropout))
            _attention_layer.append(GraAttenLayer(dim_model, c(attn), c(gcn), dropout))

        self.gconv_input = _gconv_input
        self.gconv_cond = _gconv_cond
        self.gconv_layers = nn.ModuleList(_gconv_layers)
        # self.gconv_layers1= nn.ModuleList(_gconv_layers1)
        self.atten_layers = nn.ModuleList(_attention_layer)
        self.part_convs = nn.ModuleList([PartBasedGraphConv(hid_dim) for _ in range(num_layers)])

        self.gconv_output = ChebConv(in_c=dim_model, out_c=3, K=2)
        self.fusion = refine(3)

        self.directlinear = nn.Linear(3, hid_dim)  # ¹Ç÷À·½Ïò±àÂë
        self.lengthlinear = nn.Linear(1, hid_dim)  # ¹Ç÷À³¤¶È±àÂë
        self.shrink = nn.Conv1d(hid_dim, hid_dim, 1)

        self.joint_embedding = nn.Linear(2, hid_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_pts, hid_dim))  # Î»ÖÃÇ¶Èë+×éÇ¶Èë
        self.joint_output = nn.Linear(hid_dim, 3)

        # trunc_normal_(self.pos_embed, std=.02)
        # self.apply(self._init_weights)

        drop_path_rate = 0.2
        dpr = [x.item() for x in torch.linspace(0.1, drop_path_rate, num_layers)]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def forward(self, x_2d, x_3d_dir, x_3d_bone, mask, t, cam=None):
        # x_3d = torch.cat((x_3d_dir, x_3d_bone), dim=-1)
        if self.is_train:
            B, F, J, _ = x_2d.shape
            BF = x_2d.reshape(-1, J, 2).shape[0]

            dir_emb = self.directlinear(x_3d_dir.view(BF, J, -1))  # [BF, hid_dim]
            bone_emb = self.lengthlinear(x_3d_bone.view(BF, J, -1))  # [BF, hid_dim]
            fused_3d = self.shrink((dir_emb * bone_emb).permute(0, 2, 1)).permute(0, 2, 1).view(BF, J,
                                                                                                -1)  # BC-unsqu-conv1, [BF, hid_dim] # [BF, hid_dim]

            # x = torch.cat((x_2d, fused_3d), dim=-1)
            x = x_2d
            x = x.reshape(-1, J, 2)
            x = self.gconv_input(x, self.adj)
            # x = self.joint_embedding(x)
            _, J, C = x.shape

            time_embed = self.time_mlp(t)[:, None, None, :].repeat(1, F, J, 1)
            x_2d = x_2d.reshape(-1, J, 2)
            cond_embed = self.gconv_cond(x_2d, self.adj).reshape(B, F, J, C)
            c = time_embed + cond_embed
            c = c.reshape(BF, J, C)
        else:
            x_2d_h = x_2d[:, None].repeat(1, x_3d_dir.shape[1], 1, 1, 1)
            B, H, F, J, _ = x_2d_h.shape
            BHF = x_2d_h.reshape(-1, J, 2).shape[0]

            dir_emb = self.directlinear(x_3d_dir.view(BHF, J, -1))  # [BF, hid_dim]
            bone_emb = self.lengthlinear(x_3d_bone.view(BHF, J, -1))  # [BF, hid_dim]
            fused_3d = self.shrink((dir_emb * bone_emb).permute(0, 2, 1)).permute(0, 2, 1).view(BHF, J, -1)

            # x = torch.cat((x_2d_h, fused_3d), dim=-1)
            x = x_2d_h
            x = x.reshape(-1, J, 2)
            x = self.gconv_input(x, self.adj)
            # x = self.joint_embedding(x)
            _, J, C = x.shape

            time_embed = self.time_mlp(t)[:, None, None, None, :].repeat(1, H, F, J, 1)
            x_2d_h = x_2d_h.reshape(-1, J, 2)
            cond_embed = self.gconv_cond(x_2d_h, self.adj).reshape(B, H, F, J, C)
            c = time_embed + cond_embed
            c = c.reshape(BHF, J, C)

        for i in range(self.n_layers):
            x = self.atten_layers[i](x, fused_3d, self.src_mask, mask, c)
            # x = self.gconv_layers1[i](x)
            x = self.part_convs[i](x)  # ÐÂÔö: ²¿Î»¼¶Í¼¾í»ý´¦Àí
            x = self.gconv_layers[i](x)

        x = self.gconv_output(x, self.adj)
        # x = self.joint_output(x)

        if self.is_train:
            x = x.reshape(B, F, J, -1)
            return x
        else:
            x = x.reshape(B, H, F, J, -1)
            return x


import math
import warnings


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
