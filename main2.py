import os
import glob
import time
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from common.opt import opts
from common.utils import *
from common.graph_utils import *
from common.camera import project_to_2d
from common.load_data_hm36 import Fusion
from common.h36m_dataset import Human36mDataset
from model.diffusionpose7 import DRPose as Model  # dipose6-->c3

opt = opts().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

boneindextemp = opt.boneindex_h36m.split(',')
boneindex = []
for i in range(0, len(boneindextemp), 2):
    boneindex.append([int(boneindextemp[i]), int(boneindextemp[i + 1])])


def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)


def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model)


def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    loss_all = {'loss': AccumLoss()}
    action_error_sum_j_best = define_error_list(actions)
    action_error_sum_j_avg = define_error_list(actions)
    action_error_sum_p_best = define_error_list(actions)
    action_error_sum_p_avg = define_error_list(actions)

    if split == 'train':
        model.train()
    else:
        model.eval()

    for i, data in enumerate(tqdm(dataLoader, 0, ncols=80)):
        if split == 'test' and i < 40:
            continue
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box])

        out_target = gt_3D.clone()
        inputs_traj = gt_3D[:, :, :1].clone()
        out_target[:, :, 0] = 0

        if split == 'train':
            if epoch < 10:
                mask = None
            else:
                mask = None

            output_3D = model(input_2D, out_target, mask)
        else:
            input_2D_non_flip = input_2D[:, 0]
            input_2D_flip = input_2D[:, 1]
            output_3D = model(input_2D_non_flip, out_target, gt_3D, input_2D_flip, mask=None, batch_cam=batch_cam)  # diffusion4.py
            # output_3D = model(input_2D_non_flip, out_target, input_2D_flip, mask=None,
            #                   camera_params=batch_cam, inputs_traj=inputs_traj)

        if split == 'train':
            w_mpjpe = torch.tensor([1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]).cuda()
            loss_3d_pos = weighted_mpjpe(output_3D, out_target, w_mpjpe)

            # get bone length
            inputs_3d_length = getbonelength(out_target, boneindex).mean(1)
            predicted_3d_length = getbonelength(output_3D, boneindex).mean(1)
            loss_length = opt.wl * torch.pow(inputs_3d_length - predicted_3d_length, 2).mean()

            # get bone dir
            inputs_3d_bonedir = getbonedirect(out_target, boneindex)
            predicted_bonedir = getbonedirect(output_3D, boneindex)
            loss_dir = opt.wd * torch.pow(inputs_3d_bonedir - predicted_bonedir, 2).sum(3).mean()

            loss = loss_3d_pos + loss_length + loss_dir
            # loss = loss_3d_pos

            # loss = mpjpe_cal(output_3D, out_target) + loss_length + loss_dir

            N = input_2D.size(0)
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif split == 'test':
            output_3D = output_3D[:, :, :, opt.pad].unsqueeze(3)
            output_3D[:, :, :, :, 0, :] = 0

            action_error_sum_p_avg = test_calculation_diffu(output_3D, out_target, action,
                                                            action_error_sum_p_avg, opt.dataset,
                                                            subject, opt.train, mode='p_avg')
            if not opt.train:
                # 2d reprojection
                b_sz, t_sz, h_sz, f_sz, j_sz, c_sz = output_3D.shape
                inputs_traj_single_all = inputs_traj.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)
                predicted_3d_pos_abs_single = output_3D + inputs_traj_single_all
                predicted_3d_pos_abs_single = predicted_3d_pos_abs_single.reshape(b_sz * t_sz * h_sz * f_sz, j_sz, c_sz)
                cam_single_all = batch_cam.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, f_sz,
                                                                                         1).reshape(
                    b_sz * t_sz * h_sz * f_sz, -1)
                reproject_2d = project_to_2d(predicted_3d_pos_abs_single, cam_single_all)
                reproject_2d = reproject_2d.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, 2)

                action_error_sum_p_best = test_calculation_diffu(output_3D, out_target, action, action_error_sum_p_best,
                                                                 opt.dataset, subject, opt.train, mode='p_best')
                action_error_sum_j_avg = test_calculation_diffu(output_3D, out_target, action, action_error_sum_j_avg,
                                                                opt.dataset, subject, opt.train, mode='j_avg',
                                                                reproject_2d=reproject_2d, input_2D=input_2D_non_flip)
                action_error_sum_j_best = test_calculation_diffu(output_3D, out_target, action, action_error_sum_j_best,
                                                                 opt.dataset, subject, opt.train, mode='j_best')

    if split == 'train':
        return loss_all['loss'].avg
    elif split == 'test':

        p1_p_avg, p2_p_avg = print_error(opt.dataset, action_error_sum_p_avg, opt.train, mode='p_avg')
        if not opt.train:
            p1_p_best, p2_p_best = print_error(opt.dataset, action_error_sum_p_best, opt.train, mode='p_best')
            p1_j_avg, p2_j_avg = print_error(opt.dataset, action_error_sum_j_avg, opt.train, mode='j_avg')
            p1_j_best, p2_j_best = print_error(opt.dataset, action_error_sum_j_best, opt.train, mode='j_best')
            return p1_p_avg, p2_p_avg, p1_p_best, p2_p_best, p1_j_avg, p2_j_avg, p1_j_best, p2_j_best

        return p1_p_avg, p2_p_avg


if __name__ == '__main__':
    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    print("lr: ", opt.lr)
    print("batch_size: ", opt.batch_size)
    print("channel: ", opt.channel)
    print('timestep: ', opt.timestep)
    print('samplimg_timestep: ', opt.samplimg_timestep)
    print('num_proposals: ', opt.num_proposals)
    print("GPU: ", opt.gpu)

    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)
    else:
        logtime = time.strftime('%m%d_%H%M_%S')
        log_name = 'test_' + logtime + '.log'
        logging.basicConfig(format='%(message)s', filename=os.path.join(opt.checkpoint, log_name), level=logging.INFO)

    root_path = opt.root_path
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'

    dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions(opt.actions)
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    adj = adj_mx_from_skeleton(dataset.skeleton())

    if opt.train:
        train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)

    test_data = Fusion(opt=opt, train=False, dataset=dataset, root_path=root_path)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                  shuffle=False, num_workers=int(opt.workers), pin_memory=True)

    # model = Model(opt, joints_left=joints_left, joints_right=joints_right, is_train=True).cuda()
    # model_val = Model(opt, joints_left=joints_left, joints_right=joints_right, is_train=False,
    #                   num_proposals=opt.num_proposals, sampling_timesteps=opt.samplimg_timestep).cuda()
    # HTnet
    model = Model(opt, adj, joints_left=joints_left, joints_right=joints_right, is_train=True).cuda()
    model_val = Model(opt, adj, joints_left=joints_left, joints_right=joints_right, is_train=False,
                      num_proposals=opt.num_proposals, sampling_timesteps=opt.samplimg_timestep).cuda()

    model_dict = model.state_dict()

    if opt.previous_dir != '':
        model_paths = sorted(glob.glob(os.path.join(opt.previous_dir, '*.pth')))

        for path in model_paths:
            if path.split('/')[-1].startswith('model'):
                model_path = path
                print(model_path)
        pre_dict = torch.load(model_path)

        model_dict = model.state_dict()
        state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params / 1000000)

    all_param = []
    lr = opt.lr
    all_param += list(model.parameters())

    optimizer = optim.Adam(all_param, lr=opt.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'min', factor=0.317, patience=5, verbose=True)

    for epoch in range(1, opt.nepoch):
        if opt.train:
            loss = train(opt, actions, train_dataloader, model, optimizer, epoch)

            model_val.load_state_dict(model.state_dict(), strict=False)
            p1, p2 = val(opt, actions, test_dataloader, model_val)
        else:
            model_val.load_state_dict(model.state_dict(), strict=False)
            p1_p_avg, p2_p_avg, p1_p_best, p2_p_best, p1_j_avg, p2_j_avg, p1_j_best, p2_j_best = (
                val(opt, actions, test_dataloader, model_val))

        if opt.train:
            # save_model_epoch(opt.checkpoint, epoch, model)
            if p1 < opt.previous_best_threshold:
                opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p1, model)
                opt.previous_best_threshold = p1

        if opt.train == 0:
            logging.info(
                'p_avg p1: %.2f, p2: %.2f \np_best p1: %.2f, p2: %.2f \nj_avg p1: %.2f, p2: %.2f \nj_best p1: %.2f, p2: %.2f' % (
                    p1_p_avg, p2_p_avg, p1_p_best, p2_p_best, p1_j_avg, p2_j_avg, p1_j_best, p2_j_best))
            print(
                'p_avg p1: %.2f, p2: %.2f \np_best p1: %.2f, p2: %.2f \nj_avg p1: %.2f, p2: %.2f \nj_best p1: %.2f, p2: %.2f' % (
                    p1_p_avg, p2_p_avg, p1_p_best, p2_p_best, p1_j_avg, p2_j_avg, p1_j_best, p2_j_best))
            break
        else:
            logging.info('epoch: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, p1, p2))
            print('e: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, p1, p2))

        if epoch % opt.large_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] *= opt.lr_decay
        #     lr *= opt.lr_decay

    print(opt.checkpoint)







