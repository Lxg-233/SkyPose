import math
import random
from typing import List
from collections import namedtuple
from common.opt import opts

import torch
import torch.nn.functional as F
from torch import nn

from model2.trans import GraFormer
# from model.GraFormer2 import *

__all__ = ["DRPose"]

ModelPrediction = namedtuple('ModelPrediction',
                             ['pred_noise_dir',
                              'pred_noise_bone', 'pred_x_start'])
opt = opts().parse()
boneindextemp = opt.boneindex_h36m.split(',')
boneindex = []
for i in range(0, len(boneindextemp), 2):
    boneindex.append([int(boneindextemp[i]), int(boneindextemp[i + 1])])

direct_pairs = [
    (0, 1), (1, 2), (2, 3),  # ÓÒÍÈ
    (0, 4), (4, 5), (5, 6),  # ×óÍÈ
    (0, 7), (7, 8), (8, 9), (9, 10),  # ¼¹Öù
    (8, 11), (11, 12), (12, 13),  # ÓÒ±Û
    (8, 14), (14, 15), (15, 16)  # ×ó±Û
]

symmetric_pairs = [(1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16)]

# ¶¨ÒåÉíÌå²¿Î»·Ö×é
body_parts = {
    'right_leg': [1, 2, 3],
    'left_leg': [4, 5, 6],
    'spine': [0, 7, 8, 9, 10],
    'right_arm': [11, 12, 13],
    'left_arm': [14, 15, 16]
}
joint_hierarchy = {
    0: 1,
    1: 2, 4: 2, 7: 2,
    2: 3, 5: 3, 8: 3,
    3: 4, 6: 4, 9: 4, 11: 4, 14: 4,
    10: 5, 12: 5, 15: 5,
    13: 6, 16: 6
}

from common.camera import project_to_2d


def adaptive_diffusion_step_with_2d_constraint(x_dir, x_bone, pred_noise_dir, pred_noise_bone,
                                               alpha, alpha_next, sigma, c, joint_hierarchy,
                                               preds_all_pos, inputs_2d, cam_params=None, inputs_traj=None,
                                               current_iteration=None, total_iterations=None):
    """
    »ùÓÚ2DÖØÍ¶Ó°Ô¼ÊøµÄ×ÔÊÊÓ¦À©É¢²½³¤
    Ê¹ÓÃÖ±½ÓÔ¼Êø·½·¨£¬²»Ê¹ÓÃÌÝ¶È¼ÆËã
    """
    batch_size, proposals, frames, joints, _ = x_dir.shape
    device = x_dir.device

    # 1. ¼ÆËã2DÖØÍ¶Ó°ºÍ¹Ç÷À²îÒì
    bone_dir_constraint = torch.zeros_like(x_dir)  # [B, H, F, J, 3]
    bone_length_constraint = torch.zeros_like(x_bone)  # [B, H, F, J, 1]

    if preds_all_pos is not None and inputs_2d is not None:
        # ½âÎöÔ¤²âµÄ3D×ËÌ¬Î¬¶È [Batch, Iterations, Hypotheses, Frames, Joints, Coords]
        b_sz, k_sz, h_sz, t_sz, j_sz, c_sz = preds_all_pos.shape  # [B, K, H, T, J, C]

        # È¡×îºóÒ»´Îµü´úµÄÔ¤²â½á¹û
        current_preds = preds_all_pos[:, -1]  # [B, H, T, J, C] - È¡×îºóÒ»¸öK

        # À©Õ¹¹ì¼£ÐÅÏ¢µ½ËùÓÐ¼ÙÉè
        inputs_traj_single_all = inputs_traj.unsqueeze(1).repeat(1, h_sz, 1, 1, 1)  # [B, H, T, 1, 3]

        # Ìí¼Ó¹ì¼£ÐÅÏ¢µÃµ½¾ø¶Ô3DÎ»ÖÃ
        predicted_3d_pos_abs = current_preds + inputs_traj_single_all  # [B, H, T, J, C]

        # ÖØËÜÎªÅúÁ¿´¦Àí¸ñÊ½
        predicted_3d_pos_abs_flat = predicted_3d_pos_abs.reshape(b_sz * h_sz * t_sz, j_sz, c_sz)  # [B*H*T, J, C]

        # À©Õ¹Ïà»ú²ÎÊýµ½ËùÓÐ¼ÙÉèºÍÊ±¼äÖ¡
        cam_single_all = cam_params.unsqueeze(1).unsqueeze(1).repeat(1, h_sz, t_sz, 1).reshape(b_sz * h_sz * t_sz,
                                                                                               -1)  # [B*H*T, cam_dim]

        # Í¶Ó°3D×ËÌ¬µ½2D
        reproject_2d = project_to_2d(predicted_3d_pos_abs_flat, cam_single_all)  # [B*H*T, J, 2]
        reproject_2d = reproject_2d.reshape(b_sz, h_sz, t_sz, j_sz, 2)  # [B, H, T, J, 2]

        # ¼ÆËãÊäÈë2D×ËÌ¬µÄ¹Ç÷ÀÌØÕ÷
        input_bone_dirs, input_bone_lens = compute_2d_bone_features(inputs_2d, boneindex)  # [B, T, num_bones, 2/1]

        # 3. ¶ÔÃ¿¸ö¼ÙÉè¼ÆËã¹Ç÷À²îÒì²¢Ö±½Ó×÷ÎªÔ¼ÊøÏî
        for h in range(proposals):
            current_2d = reproject_2d[:, h]  # [B, F, J, 2]

            # ¼ÆËãÖØÍ¶Ó°2DµÄ¹Ç÷ÀÌØÕ÷
            proj_bone_dirs, proj_bone_lens = compute_2d_bone_features(current_2d, boneindex)  # [B, F, num_bones, 2/1]

            input_dirs_norm = F.normalize(input_bone_dirs, dim=-1)  # [B, F, num_bones, 2]
            proj_dirs_norm = F.normalize(proj_bone_dirs, dim=-1)  # [B, F, num_bones, 2]
            cosine_sim = torch.sum(input_dirs_norm * proj_dirs_norm, dim=-1)  # [B, F, num_bones]
            # direction_loss = torch.mean(1 - cosine_sim)  # ÓàÏÒ¾àÀë
            # length_loss = F.mse_loss(proj_bone_lens, input_bone_lens)

            bone_dir_diff = (1 - cosine_sim.unsqueeze(-1)) * input_dirs_norm  # [B, F, num_bones, 2]
            bone_len_diff = torch.pow(proj_bone_lens - input_bone_lens, 2)  # [B, F, num_bones, 1]

            bone_dir_constraint[:, h, :, :, :2] = bone_dir_diff  # Ö»Ìî³äxy·ÖÁ¿
            bone_dir_constraint[:, h, :, :, 2] = 0  # z·ÖÁ¿ÉèÎª0
            bone_length_constraint[:, h, :, :, :] = bone_len_diff

    noise_dir = torch.randn_like(x_dir)
    noise_bone = torch.randn_like(x_bone)

    # 5. DDIM¸
    if bone_dir_constraint.abs().sum() > 0:  # È·±£Ô¼Êø²»È«ÎªÁã
        # ¼ÆËãÔ¼ÊøµÄÔ­Ê¼·ù¶È£¨Õâ¾ÍÊÇloss£©
        raw_magnitude_dir = torch.norm(bone_dir_constraint, dim=-1, keepdim=True)
        raw_magnitude_bone = torch.norm(bone_length_constraint, dim=-1, keepdim=True)

        # Ê¹ÓÃmax¡¢min¡¢mean½áºÏtanh¹éÒ»»¯µ½[-1, 1]
        # ¼ÆËãµ±Ç°Åú´ÎµÄÍ³¼ÆÁ¿
        max_mag_dir = raw_magnitude_dir.max()
        min_mag_dir = raw_magnitude_dir.min()
        mean_mag_dir = raw_magnitude_dir.mean()

        max_mag_bone = raw_magnitude_bone.max()
        min_mag_bone = raw_magnitude_bone.min()
        mean_mag_bone = raw_magnitude_bone.mean()

        # ±ê×¼»¯µ½ÒÔmeanÎªÖÐÐÄµÄ·Ö²¼
        if max_mag_dir > min_mag_dir + 1e-6:
            # ±ê×¼»¯µ½[-2, 2]·¶Î§£¬ÒÔmeanÎªÖÐÐÄ£¬È»ºóÓÃtanhÑ¹Ëõµ½[-1, 1]
            normalized_dir = 2.0 * (raw_magnitude_dir - mean_mag_dir) / (max_mag_dir - min_mag_dir + 1e-8)
        else:
            normalized_dir = torch.zeros_like(raw_magnitude_dir)

        if max_mag_bone > min_mag_bone + 1e-6:
            normalized_bone = 2.0 * (raw_magnitude_bone - mean_mag_bone) / (max_mag_bone - min_mag_bone + 1e-8)
        else:
            normalized_bone = torch.zeros_like(raw_magnitude_bone)

        # Ê¹ÓÃtanh½øÒ»²½Æ½»¬²¢È·±£ÔÚ[-1, 1]·¶Î§ÄÚ
        constraint_normalized_dir = torch.tanh(normalized_dir)
        constraint_normalized_bone = torch.tanh(normalized_bone)

        # ¹Ø¼ü£º·´Ïò¿ØÖÆsigma
        # constraint_normalizedÔÚ[-1, 1]·¶Î§ÄÚ
        # lossÐ¡ -> constraint_normalized½Ó½ü-1 -> scale½Ó½ü2.0 -> sigmaÔö´ó -> È¥Ôë¼õÈõ
        # loss´ó -> constraint_normalized½Ó½ü+1 -> scale½Ó½ü0.2 -> sigma¼õÐ¡ -> È¥ÔëÔöÇ¿
        constraint_scale_dir = torch.clamp( constraint_normalized_dir, 0.3, 1.5)
        constraint_scale_bone = torch.clamp(3* constraint_normalized_bone, 0.3, 1.5)

        combined_noise_dir = noise_dir
        combined_noise_bone = noise_bone

        # Ó¦ÓÃ×ÔÊÊÓ¦È¥ÔëÇ¿¶È
        adaptive_sigma_dir = sigma * constraint_scale_dir
        adaptive_sigma_bone = sigma * constraint_scale_bone

        # ¸üÐÂDDIM²½Öè
        img_dir_t = (x_dir * alpha_next.sqrt() +
                     c * pred_noise_dir +
                     adaptive_sigma_dir * combined_noise_dir)

        img_bone_t = (x_bone * alpha_next.sqrt() +
                      c * pred_noise_bone +
                      adaptive_sigma_bone * combined_noise_bone)

    else:
        # Èç¹ûÃ»ÓÐÔ¼Êø£¬Ê¹ÓÃÔ­Ê¼µÄDDIM²½Öè
        img_dir_t = (x_dir * alpha_next.sqrt() +
                     c * pred_noise_dir +
                     sigma * noise_dir)

        img_bone_t = (x_bone * alpha_next.sqrt() +
                      c * pred_noise_bone +
                      sigma * noise_bone)

    return img_dir_t, img_bone_t


def compute_2d_bone_features(poses_2d, boneindex):
    """
    ¼ÆËã2D×ËÌ¬µÄ¹Ç÷À·½ÏòºÍ³¤¶È

    Args:
        poses_2d: [B, F, J, 2]
        boneindex: ¹Ç÷ÀË÷ÒýÁÐ±í [(parent, child), ...]

    Returns:
        bone_directions: [B, F, num_bones, 2] - ¹éÒ»»¯µÄ¹Ç÷À·½Ïò
        bone_lengths: [B, F, num_bones, 1] - ¹Ç÷À³¤¶È
    """
    batch_size, frames, joints, _ = poses_2d.shape
    bone_directions = []
    bone_lengths = []

    for parent, child in boneindex:
        if parent < joints and child < joints:
            # ¼ÆËã¹Ç÷ÀÏòÁ¿
            bone_vec = poses_2d[:, :, child] - poses_2d[:, :, parent]  # [B, F, 2]

            # ¼ÆËã³¤¶È
            bone_len = torch.norm(bone_vec, dim=-1, keepdim=True)  # [B, F, 1]
            bone_lengths.append(bone_len)

            # ¼ÆËã¹éÒ»»¯·½Ïò
            bone_dir = bone_vec / (bone_len + 1e-8)  # [B, F, 2]
            bone_directions.append(bone_dir)

    bone_directions = torch.stack(bone_directions, dim=2)  # [B, F, num_bones, 2]
    bone_lengths = torch.stack(bone_lengths, dim=2)  # [B, F, num_bones, 1]

    return bone_directions, bone_lengths


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def getbonedirect(seq, boneindex):
    bs = seq.size(0)
    ss = seq.size(1)
    seq = seq.view(-1, seq.size(2), seq.size(3))
    bone = []
    for index in boneindex:
        bone.append(seq[:, index[1]] - seq[:, index[0]])
    bonedirect = torch.stack(bone, 1)
    bonesum = torch.pow(torch.pow(bonedirect, 2).sum(2), 0.5).unsqueeze(2)
    bonedirect = bonedirect / bonesum
    bonedirect = bonedirect.view(bs, ss, -1, 3)
    return bonedirect


def getbonedirect_test(seq, boneindex):
    bone = []
    for index in boneindex:
        bone.append(seq[:, :, :, index[1]] - seq[:, :, :, index[0]])
    bonedirect = torch.stack(bone, 3)
    bonesum = torch.pow(torch.pow(bonedirect, 2).sum(-1), 0.5).unsqueeze(-1)
    bonedirect = bonedirect / bonesum
    return bonedirect


def getbonelength(seq, boneindex):
    bs = seq.size(0)
    ss = seq.size(1)
    seq = seq.view(-1, seq.size(2), seq.size(3))
    bone = []
    for index in boneindex:
        bone.append(seq[:, index[1]] - seq[:, index[0]])
    bone = torch.stack(bone, 1)
    bone = torch.pow(torch.pow(bone, 2).sum(2), 0.5)
    bone = bone.view(bs, ss, bone.size(1), 1)
    return bone


def getbonelength_test(seq, boneindex):
    bone = []
    for index in boneindex:
        bone.append(seq[:, :, :, index[1]] - seq[:, :, :, index[0]])
    bone = torch.stack(bone, 3)
    bone = torch.pow(torch.pow(bone, 2).sum(-1), 0.5).unsqueeze(-1)

    return bone


class DRPose(nn.Module):
    """
    Implement DDHPose
    """

    def __init__(self, args, adj,joints_left, joints_right, is_train=True, num_proposals=1, sampling_timesteps=1):
        super().__init__()

        self.frames = args.frames
        self.num_proposals = num_proposals
        self.flip = args.test_time_augmentation
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.is_train = is_train

        # build diffusion
        timesteps = args.timestep
        # timesteps_eval = args.timestep_eval
        sampling_timesteps = sampling_timesteps
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        # self.num_timesteps_eval = int(timesteps_eval)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = args.scale
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Build Dynamic Head.
        # self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        drop_path_rate = 0
        if is_train:
            drop_path_rate = 0.1
        self.adj = adj

        # self.dir_bone_estimator = MixSTE2(num_frame=self.frames, num_joints=17, in_chans=2, embed_dim_ratio=args.cs,
        #                                   depth=args.dep,
        #                                   num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
        #                                   drop_path_rate=drop_path_rate, is_train=is_train)
        self.dir_bone_estimator = GraFormer(args, self.adj, is_train=is_train)

        self.adaptive_diffusion = getattr(args, 'adaptive_diffusion', True)


    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions_dir_bone(self, x_dir, x_bone, inputs_2d,mask, input_2d_flip, t):
        x_t_dir = torch.clamp(x_dir, min=-1.1 * self.scale, max=1.1 * self.scale)
        x_t_dir = x_t_dir / self.scale
        x_t_bone = torch.clamp(x_bone, min=-1.1 * self.scale, max=1.1 * self.scale)
        x_t_bone = x_t_bone / self.scale

        pred_pose = self.dir_bone_estimator(inputs_2d, x_t_dir, x_t_bone,mask, t)

        # input 2d flip
        x_t_dir_flip = x_t_dir.clone()
        x_t_dir_flip[:, :, :, :, 0] *= -1
        x_t_dir_flip[:, :, :, self.joints_left + self.joints_right] = x_t_dir_flip[:, :, :,
                                                                      self.joints_right + self.joints_left]
        x_t_bone_flip = x_t_bone.clone()
        x_t_bone_flip[:, :, :, self.joints_left + self.joints_right] = x_t_bone_flip[:, :, :,
                                                                       self.joints_right + self.joints_left]

        pred_pose_flip = self.dir_bone_estimator(input_2d_flip, x_t_dir_flip, x_t_bone_flip,mask, t)

        pred_pose_flip[:, :, :, :, 0] *= -1
        pred_pose_flip[:, :, :, self.joints_left + self.joints_right] = pred_pose_flip[:, :, :,
                                                                        self.joints_right + self.joints_left]
        pred_pos = (pred_pose + pred_pose_flip) / 2

        x_start_dir = getbonedirect_test(pred_pos, boneindex)
        x_start_dir = x_start_dir * self.scale
        x_start_dir = torch.clamp(x_start_dir, min=-1.1 * self.scale, max=1.1 * self.scale)
        pred_noise_dir = self.predict_noise_from_start(x_dir[:, :, :, 1:, :], t, x_start_dir)

        x_start_bone = getbonelength_test(pred_pos, boneindex)
        x_start_bone = x_start_bone * self.scale
        x_start_bone = torch.clamp(x_start_bone, min=-1.1 * self.scale, max=1.1 * self.scale)
        pred_noise_bone = self.predict_noise_from_start(x_bone[:, :, :, 1:, :], t, x_start_bone)

        x_start_pos = pred_pos
        x_start_pos = x_start_pos * self.scale
        x_start_pos = torch.clamp(x_start_pos, min=-1.1 * self.scale, max=1.1 * self.scale)

        return ModelPrediction(pred_noise_dir, pred_noise_bone, x_start_pos)

    def ddim_sample_bone_dir(self, inputs_2d, inputs_3d,mask, clip_denoised=True, do_postprocess=True, input_2d_flip=None,cam_params=None):
        batch = inputs_2d.shape[0]
        jt_num = inputs_2d.shape[-2]

        dir_shape = (batch, self.num_proposals, self.frames, jt_num, 3)
        bone_shape = (batch, self.num_proposals, self.frames, jt_num, 1)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img_dir = torch.randn(dir_shape, device='cuda')
        img_bone = torch.randn(bone_shape, device='cuda')

        x_start_dir = None
        x_start_bone = None

        preds_all_pos = []

        inputs_traj = inputs_3d[:, :, :1].clone() if inputs_3d is not None else None
        iteration_count = 0

        for time, time_next in time_pairs:
            iteration_count += 1
            time_cond = torch.full((batch,), time, dtype=torch.long).cuda()
            # self_cond = x_start if self.self_condition else None

            # print("%d/%d" % (time, total_timesteps))
            preds_pos = self.model_predictions_dir_bone(img_dir, img_bone, inputs_2d, mask,input_2d_flip, time_cond)
            pred_noise_dir, pred_noise_bone, x_start_pos = preds_pos.pred_noise_dir, preds_pos.pred_noise_bone, preds_pos.pred_x_start

            x_start_dir = getbonedirect_test(x_start_pos, boneindex)
            x_start_bone = getbonelength_test(x_start_pos, boneindex)

            preds_all_pos.append(x_start_pos)

            if time_next < 0:
                img_dir = x_start_dir
                img_bone = x_start_bone
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()


            if self.adaptive_diffusion:
                current_preds = torch.stack(preds_all_pos, dim=1) if len(preds_all_pos) > 0 else None
                img_dir_t, img_bone_t = adaptive_diffusion_step_with_2d_constraint(
                    x_start_dir, x_start_bone, pred_noise_dir, pred_noise_bone,
                    alpha, alpha_next, sigma, c, joint_hierarchy,
                    current_preds, inputs_2d, cam_params, inputs_traj,
                    current_iteration=iteration_count,
                    total_iterations=sampling_timesteps
                )
            else:
                # Ô­Ê¼À©É¢²½³¤
                noise_dir = torch.randn_like(x_start_dir)
                noise_bone = torch.randn_like(x_start_bone)
                img_dir_t = (x_start_dir * alpha_next.sqrt() +
                                c * pred_noise_dir + sigma * noise_dir)
                img_bone_t = (x_start_bone * alpha_next.sqrt() +
                                 c * pred_noise_bone +
                                 sigma * noise_bone)

            img_dir[:, :, :, 1:] = img_dir_t
            img_bone[:, :, :, 1:] = img_bone_t

        return torch.stack(preds_all_pos, dim=1)

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, input_2d, input_3d, input_2d_flip=None,mask=None,batch_cam=None):

        # Prepare Proposals.
        if not self.is_train:
            pred_pose = self.ddim_sample_bone_dir(input_2d, input_3d,mask, input_2d_flip=input_2d_flip,cam_params=batch_cam)
            return pred_pose

        if self.is_train:
            x_dir, dir_noises, x_bone_length, bone_length_noises, t = self.prepare_targets(input_3d)
            x_dir = x_dir.float()
            x_bone_length = x_bone_length.float()

            t = t.squeeze(-1)

            pred_pose = self.dir_bone_estimator(input_2d, x_dir, x_bone_length,mask, t)

            return pred_pose

    def prepare_diffusion_concat(self, pose_3d):

        t = torch.randint(0, self.num_timesteps, (1,), device='cuda').long()
        noise = torch.randn(self.frames, pose_3d.shape[1], pose_3d.shape[2], device='cuda')

        x_start = pose_3d

        x_start = x_start * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1.1 * self.scale, max=1.1 * self.scale)
        x = x / self.scale

        return x, noise, t

    def prepare_diffusion_bone_dir(self, dir, bone):

        t = torch.randint(0, self.num_timesteps, (1,), device='cuda').long()
        noise_dir = torch.randn(self.frames, dir.shape[1], dir.shape[2], device='cuda')
        noise_bone = torch.randn(self.frames, bone.shape[1], bone.shape[2], device='cuda')

        x_start_dir = dir
        x_start_bone = bone

        x_start_dir = x_start_dir * self.scale
        x_start_bone = x_start_bone * self.scale

        # noise sample
        x_dir = self.q_sample(x_start=x_start_dir, t=t, noise=noise_dir)
        x_bone = self.q_sample(x_start=x_start_bone, t=t, noise=noise_bone)

        x_dir = torch.clamp(x_dir, min=-1.1 * self.scale, max=1.1 * self.scale)
        x_dir = x_dir / self.scale
        x_bone = torch.clamp(x_bone, min=-1.1 * self.scale, max=1.1 * self.scale)
        x_bone = x_bone / self.scale

        return x_dir, noise_dir, x_bone, noise_bone, t

    def prepare_targets(self, targets):
        diffused_dir = []
        noises_dir = []
        diffused_bone_length = []
        noises_bone_length = []
        ts = []

        targets_dir = torch.zeros(targets.shape[0], targets.shape[1], targets.shape[2], 3).cuda()
        targets_bone_length = torch.zeros(targets.shape[0], targets.shape[1], targets.shape[2], 1).cuda()
        dir = getbonedirect(targets, boneindex)
        bone_length = getbonelength(targets, boneindex)
        targets_dir[:, :, 1:] = dir
        targets_bone_length[:, :, 1:] = bone_length

        for i in range(0, targets.shape[0]):
            targets_per_sample_dir = targets_dir[i]
            targets_per_sample_bone_length = targets_bone_length[i]

            d_dir, d_noise_dir, d_bone_length, d_noise_bone_length, d_t = self.prepare_diffusion_bone_dir(
                targets_per_sample_dir, targets_per_sample_bone_length)  # 3D-长度+方向--扩散

            diffused_dir.append(d_dir)
            noises_dir.append(d_noise_dir)

            diffused_bone_length.append(d_bone_length)
            noises_bone_length.append(d_noise_bone_length)
            ts.append(d_t)

        return torch.stack(diffused_dir), torch.stack(noises_dir), \
            torch.stack(diffused_bone_length), torch.stack(noises_bone_length), torch.stack(ts)

