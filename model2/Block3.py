from functools import partial
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath
# from model.blocks.drop import DropPath
from model2.GCN_conv import ModulatedGraphConv
from model2.Transformer import Attention, Mlp
import torch.nn.functional as F
# X_1
rl_2joints = [2, 3]
ll_2joints = [5, 6]
la_2joints = [12, 13]
ra_2joints = [15, 16]
part_2joints = [rl_2joints, ll_2joints, la_2joints, ra_2joints]
# X_2
rl_3joints = [1, 2, 3]
ll_3joints = [4, 5, 6]
ra_3joints = [14, 15, 16]
la_3joints = [11, 12, 13]
part_3joints = [rl_3joints, ll_3joints, la_3joints, ra_3joints]


class LJC(nn.Module):
    def __init__(self, adj, dim, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.adj = adj
        self.norm_gcn1 = norm_layer(dim)
        self.gcn1 = ModulatedGraphConv(dim, 384, self.adj)
        self.gelu = nn.GELU()
        self.gcn2 = ModulatedGraphConv(384, dim, self.adj)
        self.norm_gcn2 = norm_layer(dim)

        # ÐÂÔö£º¹Ç÷À³¤¶È¸ÐÖªµÄ¹Ø½ÚÈ¨ÖØ - »ùÓÚÈËÌå¹Ø½ÚÁ¬½ÓÏÈÑé
        self.bone_length_weight = nn.Parameter(torch.ones(17))  # Ã¿¸ö¹Ø½ÚµÄ¹Ç÷À³¤¶ÈÈ¨ÖØ
        self.joint_pair_indices = [
            (0, 1), (1, 2), (2, 3),  # ÓÒÍÈ
            (0, 4), (4, 5), (5, 6),  # ×óÍÈ
            (0, 7), (7, 8), (8, 9), (9, 10),  # ¼¹Öù
            (8, 11), (11, 12), (12, 13),  # ÓÒ±Û
            (8, 14), (14, 15), (15, 16)  # ×ó±Û
        ]


    def forward(self, x_gcn, bone_features=None):
        # Ô­ÓÐµÄGCN´¦Àí
        x_gcn_processed = x_gcn + self.drop_path(self.norm_gcn2(self.gcn2(self.gelu(self.gcn1(self.norm_gcn1(x_gcn))))))

        if bone_features is not None:    #  b1
            bone_chunk = bone_features.chunk(3, -1)[0]  # Ê¹ÓÃµÚÒ»¸öchunkÓëLJC¶ÔÓ¦
            B, N, C = bone_chunk.shape

            gate = torch.sigmoid(bone_chunk.mean(dim=-1, keepdim=True) * 0.01)  # ·Ç³£Ð¡µÄËõ·Å£¬²úÉú½Ó½ü0.5µÄÃÅ¿ØÖµ

            for parent, child in self.joint_pair_indices:
                if child < N:
                    if child in [3, 6, 10, 13, 16]:
                        joint_gate = gate[:, child, :].clone() * 0.03
                        x_gcn_temp = x_gcn_processed[:, child, :].clone()
                        x_gcn_processed[:, child, :] = x_gcn_temp * (1.0 + joint_gate)

        return x_gcn_processed


class IPC(nn.Module):  # c3
    def __init__(self, dim, mlp_hidden_dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.index_1 = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]  # 6parts
        self.index_2 = [2, 3, 5, 6, 12, 13, 15, 16]
        self.gelu = nn.GELU()
        self.norm_conv1 = norm_layer(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=0, stride=3)
        self.norm_conv1_mlp = norm_layer(dim)
        self.mlp_down_1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_conv2 = norm_layer(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=2, padding=0, stride=2)
        self.norm_conv2_mlp = norm_layer(dim)
        self.mlp_down_2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # ÐÂÔö£º»ùÓÚÉíÌå²¿¼þµÄ¹Ç÷ÀÔ¼ÊøÈ¨ÖØ
        # ËÄ¸öÖ«Ìå£ºÓÒÍÈ¡¢×óÍÈ¡¢ÓÒ±Û¡¢×ó±Û
        self.limb_bone_weights = nn.Parameter(torch.ones(4))  # 4¸öÖ«ÌåµÄ¹Ç÷ÀÔ¼ÊøÈ¨ÖØ

        # ¶¨ÒåÖ«Ìå¹Ø½ÚÓ³Éä
        self.limb_joint_map = {
            0: [1, 2, 3],  # ÓÒÍÈ
            1: [4, 5, 6],  # ×óÍÈ
            2: [11, 12, 13],  # ÓÒ±Û
            3: [14, 15, 16]  # ×ó±Û
        }


    def forward(self, x_gcn, x_conv, bone_features=None):
        x_conv = x_conv + x_gcn

        # ·ÂÕÕLJCµÄbone_chunkÈÚºÏ·½Ê½
        if bone_features is not None:
            bone_chunk = bone_features.chunk(3, -1)[1]  # Ê¹ÓÃµÚ¶þ¸öchunkÓëIPC¶ÔÓ¦

            # 1. ¼ÆËã¹Ç÷ÀÖÃÐÅ¶ÈÈ¨ÖØ
            bone_confidence = torch.norm(bone_chunk, dim=-1, keepdim=True)  # [B, N, 1]
            bone_confidence = torch.sigmoid(bone_confidence - 1.0) * 0.08  # ¹éÒ»»¯²¢¿ØÖÆÇ¿¶È£¬±ÈLJCÉÔÐ¡

            # 2. »ùÓÚÖ«ÌåµÄ¹Ç÷ÀÔ¼Êø£ºÖ«ÌåÄÚ²¿ÌØÕ÷´«µÝ
            limb_constraints = torch.zeros_like(x_conv)

            # ¶ÔÃ¿¸öÖ«ÌåÄÚ²¿½øÐÐÌØÕ÷´«µÝ
            for limb_idx, joint_indices in self.limb_joint_map.items():
                valid_indices = [i for i in joint_indices if i < x_conv.size(1) and i < bone_chunk.size(1)]

                if len(valid_indices) > 1:
                    # Ö«ÌåÄÚ²¿£º´Ó½ü¶Ë¹Ø½ÚÏòÔ¶¶Ë¹Ø½Ú´«µÝ¹Ç÷ÀÌØÕ÷
                    root_joint = valid_indices[0]  # Ö«Ìå¸ù²¿¹Ø½Ú
                    for joint_idx in valid_indices[1:]:  # ÆäËû¹Ø½Ú
                        # ¸ù²¿¹Ø½ÚµÄ¹Ç÷ÀÌØÕ÷Ó°Ïì¸ÃÖ«ÌåµÄÆäËû¹Ø½Ú
                        limb_constraints[:, joint_idx, :] = bone_chunk[:, root_joint, :] * 0.04

            # 3. ×ÔÊÊÓ¦ÈÚºÏ£º¸ù¾Ý¹Ç÷ÀÖÃÐÅ¶Èµ÷ÕûÔ¼ÊøÇ¿¶È
            x_conv = x_conv + bone_confidence * limb_constraints

        # NOTE:Conv_1  3 joints per limb
        x_conv_1 = self.norm_conv1(x_conv)
        x_conv_1 = x_conv_1.permute(0, 2, 1)
        x_pooling_1 = x_conv_1[:, :, self.index_1]
        x_pooling_1 = self.drop_path(self.gelu(self.conv1(x_pooling_1)))

        x_pooling_1 = x_pooling_1.permute(0, 2, 1)
        x_pooling_1 = x_pooling_1 + self.drop_path(self.mlp_down_1(self.norm_conv1_mlp(x_pooling_1)))
        x_pooling_1 = x_pooling_1.permute(0, 2, 1)
        for i in range(len(part_3joints)):
            num_joints = len(part_3joints[i]) - 1
            x_conv_1[:, :, part_3joints[i][1:]] = x_pooling_1[:, :, i].unsqueeze(-1).repeat(1, 1, num_joints)
        x_conv_1 = x_conv_1.permute(0, 2, 1)

        # NOTE:Conv_2  2 joints per limb
        x_conv_2 = self.norm_conv2(x_conv)
        x_conv_2 = x_conv_2.permute(0, 2, 1)
        x_pooling_2 = x_conv_2[:, :, self.index_2]
        x_pooling_2 = self.drop_path(self.gelu(self.conv2(x_pooling_2)))

        x_pooling_2 = x_pooling_2.permute(0, 2, 1)
        x_pooling_2 = x_pooling_2 + self.drop_path(self.mlp_down_2(self.norm_conv2_mlp(x_pooling_2)))
        x_pooling_2 = x_pooling_2.permute(0, 2, 1)
        for i in range(len(part_2joints)):
            num_joints = len(part_2joints[i]) - 1
            x_conv_2[:, :, part_2joints[i][1:]] = x_pooling_2[:, :, i].unsqueeze(-1).repeat(1, 1, num_joints)
        x_conv_2 = x_conv_2.permute(0, 2, 1)

        x_conv = x_conv_1 + x_conv_2 + x_conv
        return x_conv




class GBI(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, length=1):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_attn = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, length=length)

        # ÐÂÔö£º¹Ç÷ÀÌØÕ÷½»²æ×¢ÒâÁ¦
        self.norm_bone = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
                                         qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x_conv, x_attn, bone_features=None):
        x_attn = x_attn + x_conv

        bone_chunks = bone_features.chunk(3, -1)  # ·Ö³É3¸öchunk
        bone_chunk = bone_chunks[2]

        if bone_chunk.size(-1) != x_attn.size(-1):
            if not hasattr(self, 'bone_proj'):
                self.bone_proj = nn.Linear(bone_chunk.size(-1), x_attn.size(-1)).to(bone_chunk.device)
            bone_chunk = self.bone_proj(bone_chunk)

        global_bone_info = bone_chunk.mean(dim=1, keepdim=True)  # [B, 1, C]

        norm_attn = torch.nn.functional.normalize(x_attn, dim=-1)
        norm_bone = torch.nn.functional.normalize(global_bone_info, dim=-1)

        bone_relevance = torch.bmm(norm_attn, norm_bone.transpose(1, 2))  # [B, J, 1]
        bone_relevance = torch.sigmoid(bone_relevance * 0.5)  # Ëõ·Å²¢ÏÞÖÆÔÚ[0,1]

        bone_modulation = self.norm_bone(bone_chunk) * 0.02  # ¼«Ð¡ÏµÊý
        bone_enhanced = bone_relevance * bone_modulation

        x_attn2 = x_attn + bone_enhanced

        # x_attn = x_attn + self.drop_path(self.cross_attn(self.norm_attn(x_attn),x_attn2,x_attn2))
        # x_attn = x_attn + self.drop_path(self.cross_attn( x_attn2, x_attn2,x_attn2))  #c2
        x_attn = x_attn + self.drop_path(self.attn(x_attn+bone_chunk))  # c3
        # x_attn = x_attn + self.drop_path(self.cross_attn((x_attn + bone_chunk + x_conv),x_attn,x_attn))
        # 1. IPC整体修改
        # 2. GBI cross----->c3
        # x_attn = x_attn + self.drop_path(self.cross_attn(self.norm_attn(x_attn), self.norm_attn(x_attn), x_attn2))

        return x_attn



class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape

        # Query from the main features
        q = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Hiremixer(nn.Module):
    def __init__(self, adj, depth=8, embed_dim=512, mlp_hidden_dim=1024, h=8, drop_rate=0.1, length=9):
        super().__init__()
        drop_path_rate = 0.3
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0.1, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                adj, dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, length=length)
            for i in range(depth)])
        self.Temporal_norm = norm_layer(embed_dim)

    def forward(self, x, bone_features=None):
        for blk in self.blocks:
            x = blk(x, bone_features)
        x = self.Temporal_norm(x)
        return x


class Block(nn.Module):
    def __init__(self, adj, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, length=1):
        super().__init__()

        dim = int(dim / 3)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Three sub-modules
        self.lgc = LJC(adj, dim, drop_path=drop_path, norm_layer=nn.LayerNorm)
        self.ipc = IPC(dim, mlp_hidden_dim, drop=drop, drop_path=drop_path, act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.gbi = GBI(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0.1, attn_drop=attn_drop,
                       drop_path=drop_path, norm_layer=nn.LayerNorm, length=length)

        self.norm_mlp = norm_layer(dim * 3)
        self.mlp = Mlp(in_features=dim * 3, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, bone_features=None):
        # Ê¹ÓÃclone´´½¨ÊäÈëµÄ¸±±¾£¬·ÀÖ¹¾ÍµØ²Ù×÷Ó°ÏìÔ­Ê¼ÊäÈë
        x_split = torch.chunk(x.clone(), 3, -1)
        x_lgc, x_ipc, x_gbi = x_split

        # ¸Ä½ø£º¸üÓÐÑ¡ÔñÐÔµØÊ¹ÓÃ¹Ç÷ÀÌØÕ÷
        # 1. Ê×ÏÈ¼ì²é¹Ç÷ÀÌØÕ÷µÄÓÐÐ§ÐÔ
        # valid_bone_features = None
        # if bone_features is not None:
        #     # Èç¹ûÓÐ¹Ç÷ÀÌØÕ÷£¬¼ÆËãÆäÓÐÐ§ÐÔµÃ·Ö
        #     bone_norm = torch.norm(bone_features, dim=-1, keepdim=True)
        #     # ¹Ç÷ÀÌØÕ÷ºÏÀíÐÔ¼ì²é£º¼«¶ËÖµ¹ýÂË£¨Èç¹ý´ó»ò¹ýÐ¡µÄÖµÍ¨³£Ö¸Ê¾²»¿É¿¿µÄ¹Ç÷À¹À¼Æ£©
        #     validity_mask = (bone_norm > 1e-6) & (bone_norm < 100.0)
        #     # Ö»ÔÚÌØÕ÷ÓÐÐ§Ê±Ê¹ÓÃ
        #     if validity_mask.float().mean() > 0.5:  # È·±£´ó²¿·ÖÌØÕ÷ÓÐÐ§
        #         valid_bone_features = bone_features
        valid_bone_features = bone_features

        # Local Joint-level Connection (LJC) - ¸ü±£ÊØµÄ¹Ç÷ÀÌØÕ÷ÀûÓÃ
        x_lgc = self.lgc(x_lgc, valid_bone_features)

        # Inter-Part Constraint (IPC) - Ö÷Òª´¦ÀíÖ«Ìå¼ä¹ØÏµ
        x_ipc = self.ipc(x_lgc, x_ipc, valid_bone_features)

        # Global body-level Interaction (GBI) - ÇáÁ¿¼¶¹Ç÷ÀÐÅÏ¢Òýµ¼
        x_gbi = self.gbi(x_ipc, x_gbi, valid_bone_features)

        x_cat = torch.cat([x_lgc, x_ipc, x_gbi], -1)
        output = x_cat + self.drop_path(self.mlp(self.norm_mlp(x_cat)))
        return output


class Hiremixer_frame(nn.Module):
    def __init__(self, adj, depth=8, embed_dim=512, mlp_hidden_dim=1024, h=8, drop_rate=0.1, length=9):
        super().__init__()
        drop_path_rate = 0.3
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0.1, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block_frame(
                adj, dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, length=length)
            for i in range(depth)])
        self.Temporal_norm = norm_layer(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.Temporal_norm(x)
        return x


class Block_frame(nn.Module):
    def __init__(self, adj, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, length=1):
        super().__init__()

        dim = int(dim / 2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Three sub-modules
        self.lgc = LJC(adj, dim, drop_path=drop_path, norm_layer=nn.LayerNorm)
        self.gbi = GBI(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0.1, attn_drop=attn_drop,
                       drop_path=drop_path, norm_layer=nn.LayerNorm, length=length)

        self.norm_mlp = norm_layer(dim * 2)
        self.mlp = Mlp(in_features=dim * 2, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_split = torch.chunk(x, 2, -1)
        x_lgc, x_gbi = x_split
        # Local Joint-level Connection (LJC)
        x_lgc = self.lgc(x_lgc)
        # Global body-level Interaction (GBI)
        x_gbi = self.gbi(x_lgc, x_gbi)
        x_cat = torch.cat([x_lgc, x_gbi], -1)
        x = x_cat + self.drop_path(self.mlp(self.norm_mlp(x_cat)))
        return x


class Block_ipc(nn.Module):
    def __init__(self, adj, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, length=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Three sub-modules
        self.ipc = IPC(dim, mlp_hidden_dim, drop=drop, drop_path=drop_path, act_layer=nn.GELU, norm_layer=nn.LayerNorm)

        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.ipc(x)
        x = x + self.drop_path(self.mlp(self.norm_mlp(x)))
        return x