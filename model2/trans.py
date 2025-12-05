import sys
from einops.einops import rearrange
sys.path.append("..")
import torch
import torch.nn as nn
import math
# from model2.Block2 import Hiremixer      # diff67+trans2
from model2.Block2 import Hiremixer
from common.opt import opts
opt = opts().parse()


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码用于时间步"""
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





class GraFormer(nn.Module):
    def __init__(self, args, adj, is_train=True):
        super().__init__()
        self.is_train = is_train

        if args == -1:
            layers, channel, d_hid, length  = 3, 512, 1024, 27
            self.num_joints_in, self.num_joints_out = 17, 17
        else:
            layers, channel, d_hid, length  = 3, 240, 1024, 1
            self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints

        # 2D关键点嵌入
        self.patch_embed = nn.Linear(2, channel)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_joints_in, channel))

        # 骨骼方向编码器 (3D -> channel)
        self.direction_encoder = nn.Sequential(
            nn.Linear(3, channel // 4),
            nn.ReLU(),
            nn.Linear(channel // 4, channel//2)
        )

        # 骨骼长度编码器 (1D -> channel)
        self.length_encoder = nn.Sequential(
            nn.Linear(1, channel // 4),
            nn.ReLU(),
            nn.Linear(channel // 4, channel//2)
        )

        self.bone_fusion = nn.Sequential(
            nn.Linear(channel , channel *2),  # 输出3倍channel
            nn.ReLU(),
            nn.Linear(channel *2, channel)
        )
        self.shrink = nn.Conv1d(2*channel, channel//3, 1)

        # 时间步编码
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(channel),
            nn.Linear(channel, channel * 2),
            nn.SiLU(),
            nn.Linear(channel * 2, channel),
        )

        # 条件嵌入 (用于融合时间步和2D特征)
        self.cond_embed = nn.Linear(2, channel)

        # 核心处理模块
        self.Hiremixer = Hiremixer(adj, layers, channel, d_hid, length=length)

        # 输出层
        self.fcn = nn.Linear(channel, 3)




    def encode_bone_features(self, x_3d_dir, x_3d_bone, batch_format):
        """编码骨骼方向和长度特征"""
        if self.is_train:
            B, F, J, _ = batch_format
            BF = B * F

            # 编码方向和长度 [BF, J, channel]
            dir_features = self.direction_encoder(x_3d_dir.view(BF, J, -1))
            length_features = self.length_encoder(x_3d_bone.view(BF, J, -1))

        else:
            B, H, F, J, _ = batch_format
            BHF = B * H * F

            # 编码方向和长度 [BHF, J, channel]
            dir_features = self.direction_encoder(x_3d_dir.view(BHF, J, -1))
            length_features = self.length_encoder(x_3d_bone.view(BHF, J, -1))

        # 融合方向和长度特征
        combined_features = torch.cat([dir_features, length_features], dim=-1)
        bone_features = self.bone_fusion(combined_features)
        # bone_features = self.shrink(combined_features.permute(0, 2, 1)).permute(0, 2, 1)

        return bone_features

    def forward(self, x_2d, x_3d_dir, x_3d_bone, mask, t, cam=None):
        """
        前向传播
        Args:
            x_2d: 2D关键点 [B, F, J, 2] (训练) 或 [B, F, J, 2] (推理)
            x_3d_dir: 骨骼方向 [B, F, J, 3] (训练) 或 [B, H, F, J, 3] (推理)
            x_3d_bone: 骨骼长度 [B, F, J, 1] (训练) 或 [B, H, F, J, 1] (推理)
            mask: 掩码
            t: 时间步
        """

        if self.is_train:
            # 训练模式 [B, F, J, 2]
            B, F, J, _ = x_2d.shape
            BF = B * F

            # 处理2D关键点
            x = rearrange(x_2d, 'b f j c -> (b f) j c').contiguous()
            x = self.patch_embed(x) + self.pos_embed

            # 编码骨骼特征
            bone_features = self.encode_bone_features(x_3d_dir, x_3d_bone, (B, F, J, None))

            # 时间步嵌入
            time_embed = self.time_mlp(t)  # [B, channel]
            time_embed = time_embed[:, None, None, :].repeat(1, F, J, 1)  # [B, F, J, channel]
            time_embed = rearrange(time_embed, 'b f j c -> (b f) j c')  # [BF, J, channel]

            # 条件嵌入 (2D特征)
            x_2d_flat = rearrange(x_2d, 'b f j c -> (b f) j c')
            cond_embed = self.cond_embed(x_2d_flat)  # [BF, J, channel]

            # 融合所有特征 - 不直接相加，而是分别保留用于交叉注意力
            x = x + time_embed + cond_embed  # 主特征：2D + 时间 + 条件
            # bone_features 保持独立，用于交叉注意力

        else:
            # 推理模式 [B, F, J, 2] -> [B, H, F, J, 2]
            x_2d_h = x_2d[:, None].repeat(1, x_3d_dir.shape[1], 1, 1, 1)
            B, H, F, J, _ = x_2d_h.shape
            BHF = B * H * F

            # 处理2D关键点
            x = rearrange(x_2d_h, 'b h f j c -> (b h f) j c').contiguous()
            x = self.patch_embed(x) + self.pos_embed

            # 编码骨骼特征
            bone_features = self.encode_bone_features(x_3d_dir, x_3d_bone, (B, H, F, J, None))

            # 时间步嵌入
            time_embed = self.time_mlp(t)  # [B, channel]
            time_embed = time_embed[:, None, None, None, :].repeat(1, H, F, J, 1)  # [B, H, F, J, channel]
            time_embed = rearrange(time_embed, 'b h f j c -> (b h f) j c')  # [BHF, J, channel]

            # 条件嵌入
            x_2d_flat = rearrange(x_2d_h, 'b h f j c -> (b h f) j c')
            cond_embed = self.cond_embed(x_2d_flat)  # [BHF, J, channel]

            # 融合所有特征 - 不直接相加，而是分别保留用于交叉注意力
            x = x + time_embed + cond_embed  # 主特征：2D + 时间 + 条件
            # bone_features 保持独立，用于交叉注意力

        # 通过Hiremixer处理，传入骨骼特征进行交叉注意力
        # bone_features = None
        x = self.Hiremixer(x, bone_features)

        # 输出预测
        x = self.fcn(x)

        # 重塑输出
        if self.is_train:
            x = x.view(B, F, self.num_joints_out, x.shape[-1])
            return x
        else:
            x = x.view(B, H, F, self.num_joints_out, x.shape[-1])
            return x
