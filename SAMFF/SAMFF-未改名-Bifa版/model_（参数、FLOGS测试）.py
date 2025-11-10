import torch
import torch.nn as nn
import time
import os
import json
import numpy as np
from thop import profile
import models as Model
import data as Data
import argparse
import logging
import core.logger as Logger
from torch.cuda.amp import autocast
## 在BiFA-main目录下运行
#python model_（参数、FLOGS测试）.py --config config/whu.json
def count_parameters(model):
    """计算模型的参数数量（单位：百万）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def measure_time_and_memory(model, train_loader, loss_fun, optimizer, device, amp_enabled=False):
    """测量单个训练轮次的时间和内存使用情况"""
    model.train()
    
    # 预热GPU
    for _ in range(5):
        for train_data in train_loader:
            train_im1 = train_data['A'].to(device)
            train_im2 = train_data['B'].to(device)
            pred_img = model(train_im1, train_im2)
            gt = train_data['L'].to(device).long()
            train_loss = loss_fun(pred_img, gt)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            break
    
    # 同步GPU以确保所有操作完成
    torch.cuda.synchronize()
    
    # 开始计时
    start_time = time.time()
    batch_count = 0
    
    # 记录内存使用情况
    max_memory_allocated = 0
    
    for train_data in train_loader:
        train_im1 = train_data['A'].to(device)
        train_im2 = train_data['B'].to(device)
        
        if amp_enabled:
            with autocast():
                pred_img = model(train_im1, train_im2)
                gt = train_data['L'].to(device).long()
                train_loss = loss_fun(pred_img, gt)
        else:
            pred_img = model(train_im1, train_im2)
            gt = train_data['L'].to(device).long()
            train_loss = loss_fun(pred_img, gt)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # 更新最大内存使用量
        max_memory_allocated = max(max_memory_allocated, torch.cuda.max_memory_allocated())
        
        batch_count += 1
    
    # 同步GPU以确保所有操作完成
    torch.cuda.synchronize()
    
    # 计算总时间和平均每批次时间
    total_time = time.time() - start_time
    avg_time_per_batch = total_time / batch_count
    
    # 将内存使用量转换为GB
    max_memory_allocated_gb = max_memory_allocated / (1024 ** 3)
    
    return {
        'total_time': total_time,
        'avg_time_per_batch': avg_time_per_batch,
        'max_memory_allocated_gb': max_memory_allocated_gb,
        'batch_count': batch_count
    }

def calculate_flops(model, input_shape=(3, 256, 256)):
    """计算模型的浮点运算次数（单位：G）"""
    device = next(model.parameters()).device
    
    # 确保模型的所有参数都在同一设备上
    if isinstance(model, nn.DataParallel):
        # 如果是DataParallel模型，使用非DataParallel版本进行计算
        model_for_profile = model.module
    else:
        model_for_profile = model
    
    # 确保模型在正确的设备上
    model_for_profile = model_for_profile.to(device)
    
    # 创建示例输入
    x1 = torch.randn(1, input_shape[0], input_shape[1], input_shape[2]).to(device)
    x2 = torch.randn(1, input_shape[0], input_shape[1], input_shape[2]).to(device)
    
    # 使用thop计算FLOPs
    flops, params = profile(model_for_profile, inputs=(x1, x2), verbose=False)
    
    # 转换为G FLOPs
    return flops / 1e9

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/whu.json',
                        help='JSON file for configuration')
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'test'], help='Run either train(training + validation) or testing',)
    parser.add_argument('--gpu_ids', type=str, default=None)
    parser.add_argument('--eval_only', action='store_true', help='只进行评估，不进行训练')
    parser.add_argument('--input_size', type=int, default=256, help='输入图像大小')
    
    # 解析参数
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    
    # 设置日志
    Logger.setup_logger(logger_name=None, root=opt['path_cd']['log'], phase='eval',
                        level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    
    # 创建数据集和数据加载器
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'test':
            print("创建 [train] 变化检测数据加载器")
            train_set = Data.create_cd_dataset(dataset_opt=dataset_opt, phase=phase)
            train_loader = Data.create_cd_dataloader(train_set, dataset_opt, phase)
            opt['len_train_dataloader'] = len(train_loader)
            break
    
    # 设置设备
    if opt['gpu_ids'] is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{opt['gpu_ids'][0]}" if isinstance(opt['gpu_ids'], list) else "cuda:0")
        print(f"使用GPU: {device}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    
    # 创建模型
    cd_model = Model.create_CD_model(opt)
    cd_model = cd_model.to(device)
    
    # 如果使用多GPU，转换为DataParallel
    if opt['gpu_ids'] is not None and len(opt['gpu_ids']) > 0 and torch.cuda.device_count() > 1:
        cd_model = nn.DataParallel(cd_model, device_ids=[int(i) for i in opt['gpu_ids']])
        print(f"使用多GPU: {opt['gpu_ids']}")
    
    # 创建损失函数
    if opt['model']['loss'] == 'ce_dice':
        from models.loss import ce_dice as loss_fun
    elif opt['model']['loss'] == 'ce':
        from models.loss import cross_entropy as loss_fun
    
    # 创建优化器
    if opt['train']["optimizer"]["type"] == 'adam':
        optimizer = torch.optim.Adam(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"])
    elif opt['train']["optimizer"]["type"] == 'adamw':
        optimizer = torch.optim.AdamW(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"])
    elif opt['train']["optimizer"]["type"] == 'sgd':
        optimizer = torch.optim.SGD(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"],
                                    momentum=0.9, weight_decay=5e-4)
    
    # 计算参数数量
    params_count = count_parameters(cd_model)
    
    # 计算FLOPs
    input_shape = (3, args.input_size, args.input_size)
    flops = calculate_flops(cd_model, input_shape)
    
    # 测量训练时间
    if not args.eval_only and 'train_loader' in locals():
        time_results = measure_time_and_memory(cd_model, train_loader, loss_fun, optimizer, device)
    else:
        time_results = {"total_time": 0, "avg_time_per_batch": 0, "max_memory_allocated_gb": 0, "batch_count": 0}
    
    # 打印结果
    print("\n" + "="*50)
    print("模型评估结果:")
    print(f"参数数量 (Params): {params_count:.2f}M")
    print(f"浮点运算次数 (FLOPs): {flops:.2f}G")
    if not args.eval_only and 'train_loader' in locals():
        print(f"训练时间 (Time):")
        print(f"  - 总时间: {time_results['total_time']:.2f}秒")
        print(f"  - 平均每批次时间: {time_results['avg_time_per_batch']:.4f}秒")
        print(f"  - 总批次数: {time_results['batch_count']}")
        print(f"  - 最大GPU内存使用: {time_results['max_memory_allocated_gb']:.2f}GB")
    print("="*50)
    
    # 保存结果到JSON文件
    results = {
        "params_count_millions": float(params_count),
        "flops_giga": float(flops),
        "training_time_seconds": float(time_results['total_time']),
        "avg_batch_time_seconds": float(time_results['avg_time_per_batch']),
        "max_memory_allocated_gb": float(time_results['max_memory_allocated_gb']),
        "batch_count": int(time_results['batch_count']),
        "input_shape": list(input_shape)
    }
    
    os.makedirs("evaluation_results", exist_ok=True)
    with open("evaluation_results/model_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"评估结果已保存到 evaluation_results/model_metrics.json") 