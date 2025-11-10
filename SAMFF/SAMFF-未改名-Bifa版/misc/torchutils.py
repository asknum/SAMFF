import torch
import torch.nn as nn
from torch import Tensor, einsum
from torch.optim import lr_scheduler
from typing import Iterable, Set, Tuple
import logging
import os
logger = logging.getLogger('base')

def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))
    if seg.ndim == 4:
        seg = seg.squeeze(dim=1)
    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args['sheduler']['lr_policy'] == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args['n_epoch'] + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args['sheduler']['lr_policy'] == 'step':
        step_size = args['n_epoch']//args['sheduler']['n_steps']
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=args['sheduler']['gamma'])
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def save_network(opt, epoch, cd_model, optimizer, is_best_model=False, metric_score=None):
    cd_gen_path = os.path.join(
        opt['path_cd']['checkpoint'], 'cd_model_E{}_gen.pth'.format(epoch))
    cd_opt_path = os.path.join(
        opt['path_cd']['checkpoint'], 'cd_model_E{}_opt.pth'.format(epoch))

    if is_best_model:
        best_cd_gen_path = os.path.join(
            opt['path_cd']['checkpoint'], 'best_cd_model_gen.pth')
        best_cd_opt_path = os.path.join(
            opt['path_cd']['checkpoint'], 'best_cd_model_opt.pth')
        
        # 保存带epoch和得分的最佳模型检查点路径（永久保存，不会被覆盖）
        if metric_score is not None:
            score_cd_gen_path = os.path.join(
                opt['path_cd']['checkpoint'], 'best_cd_model_mF1_{:.6f}_E{}_gen.pth'.format(metric_score, epoch))
            score_cd_opt_path = os.path.join(
                opt['path_cd']['checkpoint'], 'best_cd_model_mF1_{:.6f}_E{}_opt.pth'.format(metric_score, epoch))
        else:
            # 如果没有提供得分，使用epoch信息
            score_cd_gen_path = os.path.join(
                opt['path_cd']['checkpoint'], 'best_cd_model_E{}_gen.pth'.format(epoch))
            score_cd_opt_path = os.path.join(
                opt['path_cd']['checkpoint'], 'best_cd_model_E{}_opt.pth'.format(epoch))

    # Save CD model pareamters
    network = cd_model
    if isinstance(cd_model, nn.DataParallel):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    
    # 保存当前epoch的模型
    torch.save(state_dict, cd_gen_path)
    
    if is_best_model:
        # 保存不带得分的最佳模型(覆盖旧的，用于快速加载最新最佳模型)
        torch.save(state_dict, best_cd_gen_path)
        # 保存带详细信息的最佳模型(永久保存，不会被覆盖)
        torch.save(state_dict, score_cd_gen_path)

    # Save CD optimizer paramers
    opt_state = {'epoch': epoch,
                 'scheduler': None,
                 'optimizer': None}
    opt_state['optimizer'] = optimizer.state_dict()
    
    # 保存当前epoch的优化器状态
    torch.save(opt_state, cd_opt_path)
    
    if is_best_model:
        # 保存不带得分的最佳优化器状态(覆盖旧的，用于快速加载最新最佳模型)
        torch.save(opt_state, best_cd_opt_path)
        # 保存带详细信息的最佳优化器状态(永久保存，不会被覆盖)
        torch.save(opt_state, score_cd_opt_path)

    # Print info
    logger.info(
        'Saved current CD model in [{:s}] ...'.format(cd_gen_path))
    if is_best_model:
        logger.info(
            'Saved best CD model in [{:s}] ...'.format(best_cd_gen_path))
        logger.info(
            'Saved detailed best CD model in [{:s}] ...'.format(score_cd_gen_path))
