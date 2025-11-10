import torch
import torch.optim as optim
import data as Data
import models as Model
import torch.nn as nn
import argparse
import logging
import core.logger as Logger
import os
import numpy as np
import time
from misc.metric_tools import ConfuseMatrixMeter
from models.loss import *
from collections import OrderedDict
import core.metrics as Metrics
from misc.torchutils import get_scheduler, save_network

torch.manual_seed(2023)            # 为CPU设置随机种子 2023
torch.cuda.manual_seed(2023)      # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(2023)  # 为所有GPU设置随机种子

if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='E:\PycharmProject\BiFA\config\whu.json',
                        help='JSON file for configuration')
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'test'], help='Run either train(training + validation) or testing',)
    parser.add_argument('--gpu_ids', type=str, default=None)
    parser.add_argument('-log_eval', action='store_true')

    #paser config
    args =parser.parse_args()
    opt = Logger.parse(args)

    #Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    #logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(logger_name=None, root=opt['path_cd']['log'], phase='train',
                        level=logging.INFO, screen=True)
    Logger.setup_logger(logger_name='test', root=opt['path_cd']['log'], phase='test',
                        level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    #dataset
    for phase, dataset_opt in opt['datasets'].items(): #train train{}
        #print(" phase is {}, dataopt is {}".format(phase, dataset_opt))
        if phase == 'train' and args.phase != 'test':
            print("Creat [train] change-detection dataloader")
            train_set = Data.create_cd_dataset(dataset_opt=dataset_opt, phase=phase)
            train_loader = Data.create_cd_dataloader(train_set, dataset_opt, phase)
            opt['len_train_dataloader'] = len(train_loader)

        elif phase == 'val' and args.phase != 'test':
            print("Creat [val] change-detection dataloader")
            val_set = Data.create_cd_dataset(dataset_opt=dataset_opt, phase=phase)
            val_loader = Data.create_cd_dataloader(val_set, dataset_opt, phase)
            opt['len_val_dataloader'] = len(val_loader)

        # elif phase == 'test' and args.phase == 'test':
        elif phase == 'test':
            print("Creat [test] change-detection dataloader")
            test_set = Data.create_cd_dataset(dataset_opt=dataset_opt, phase=phase)
            test_loader = Data.create_cd_dataloader(test_set, dataset_opt, phase)
            opt['len_test_dataloader'] = len(test_loader)

    logger.info('Initial Dataset Finished')

    #Create cd model
    cd_model = Model.create_CD_model(opt)

    #Create criterion
    if opt['model']['loss'] == 'ce_dice':
        loss_fun = ce_dice
    elif opt['model']['loss'] == 'ce':
        loss_fun = cross_entropy
    #loss_fun.to(opt["gpu_ids"])

    #Create optimer
    if opt['train']["optimizer"]["type"] == 'adam':
        optimer = optim.Adam(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"])
    elif opt['train']["optimizer"]["type"] == 'adamw':
        optimer = optim.AdamW(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"])
    elif opt['train']["optimizer"]["type"] == 'sgd':
        optimer = optim.SGD(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"],
                            momentum=0.9, weight_decay=5e-4)

    device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
    cd_model.to(device)
    if len(opt['gpu_ids']) > 0:
        cd_model = nn.DataParallel(cd_model)
    metric = ConfuseMatrixMeter(n_class=2)
    log_dict = OrderedDict()
    #################
    # Training loop #
    #################
    if opt['phase'] == 'train':
        best_mF1 = 0.0
        
        # 打印总体训练设置
        print('\n')
        print('='*50)
        print('训练配置信息:')
        print(f'总Epoch数: {opt["train"]["n_epoch"]}')
        print(f'训练集BatchSize: {opt["datasets"]["train"]["batch_size"]}')
        print(f'验证集BatchSize: {opt["datasets"]["val"]["batch_size"]}')
        print(f'训练集总轮数: {len(train_loader)}')
        print(f'验证集总轮数: {len(val_loader)}')
        print(f'初始学习率: {opt["train"]["optimizer"]["lr"]}')
        print(f'优化器类型: {opt["train"]["optimizer"]["type"]}')
        print(f'学习率调度策略: {opt["train"]["sheduler"]["lr_policy"]}')
        print(f'损失函数: {opt["model"]["loss"]}')
        print(f'模型: {opt["model"]["name"]}')
        print('='*50)
        print('\n')
        
        for current_epoch in range(0, opt['train']['n_epoch']):
            print("......Begin Training......")
            metric.clear()
            cd_model.train()
            train_result_path = '{}/train/{}'.format(opt['path_cd']['result'], current_epoch)
            os.makedirs(train_result_path, exist_ok=True)

            #################
            #    Training   #
            #################
            message = 'lr: %0.7f\n \n' % optimer.param_groups[0]['lr']
            logger.info(message)
            
            # 记录开始时间
            start_time = time.time()
            batch_start_time = time.time()
            
            for current_step, train_data in enumerate(train_loader):
                train_im1 = train_data['A'].to(device)
                train_im2 = train_data['B'].to(device)
                pred_img = cd_model(train_im1, train_im2)
                gt = train_data['L'].to(device).long()
                train_loss = loss_fun(pred_img, gt)
                optimer.zero_grad()
                train_loss.backward()
                optimer.step()
                log_dict['loss'] = train_loss.item()

                #pred score
                G_pred = pred_img.detach()
                G_pred = torch.argmax(G_pred, dim=1)
                current_score = metric.update_cm(pr=G_pred.cpu().numpy(), gt=gt.detach().cpu().numpy())
                log_dict['running_acc'] = current_score.item()
                
                # 每100个batch打印当前批次、用时和学习率
                if (current_step + 1) % 100 == 0:
                    batch_time = time.time() - batch_start_time
                    print(f'批次: [{current_step + 1}/{len(train_loader)}], 用时: {batch_time:.2f}秒, 学习率: {optimer.param_groups[0]["lr"]:.7f}')
                    batch_start_time = time.time()

                # log running batch status
                if current_step % opt['train']['train_print_iter'] == 0:
                    # message
                    logs = log_dict
                    message = '[Training CD]. epoch: [%d/%d]. Itter: [%d/%d], CD_loss: %.5f, running_mf1: %.5f\n' % \
                              (current_epoch, opt['train']['n_epoch'] - 1, current_step, len(train_loader), logs['loss'],
                               logs['running_acc'])
                    logger.info(message)
                    #vis
                    out_dict = OrderedDict()
                    out_dict['pred_cm'] = torch.argmax(pred_img, dim=1, keepdim=False)
                    out_dict['gt_cm'] = gt
                    visuals = out_dict

                    img_mode = "grid"
                    if img_mode == "single":
                        # Converting to uint8
                        img_A = Metrics.tensor2img(train_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        img_B = Metrics.tensor2img(train_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        gt_cm = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8,
                                                   min_max=(0, 1))  # uint8
                        pred_cm = Metrics.tensor2img(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8,
                                                     min_max=(0, 1))  # uint8

                        # save imgs
                        Metrics.save_img(
                            img_A, '{}/img_A_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(
                            img_B, '{}/img_B_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(
                            pred_cm, '{}/img_pred_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(
                            gt_cm, '{}/img_gt_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                    else:
                        # grid img
                        visuals['pred_cm'] = visuals['pred_cm'] * 2.0 - 1.0
                        visuals['gt_cm'] = visuals['gt_cm'] * 2.0 - 1.0
                        # print(isinstance(train_data['A'], torch.cuda.FloatTensor))
                        # print(isinstance(train_data['B'], torch.cuda.FloatTensor))
                        # print(isinstance(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), torch.cuda.FloatTensor))
                        # print(isinstance(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), torch.cuda.FloatTensor))
                        grid_img = torch.cat((train_data['A'].to(device),
                                              train_data['B'].to(device),
                                              visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1),
                                              visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                             dim=0)
                        grid_img = Metrics.tensor2img(grid_img)  # uint8
                        Metrics.save_img(
                            grid_img,
                            '{}/img_A_B_pred_gt_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))

            ### log epoch status ###
            scores = metric.get_scores()
            epoch_acc = scores['mf1']
            log_dict['epoch_acc'] = epoch_acc.item()
            for k, v in scores.items():
                log_dict[k] = v
            logs =log_dict
            message = '[Training CD (epoch summary)]: epoch: [%d/%d]. epoch_mF1=%.5f \n' % \
                      (current_epoch, opt['train']['n_epoch'] - 1, logs['epoch_acc'])
            for k, v in logs.items():
                message += '{:s}: {:.4e} '.format(k, v)
            message += '\n'
            logger.info(message)
            
            # 计算并打印整个epoch的训练用时
            epoch_time = time.time() - start_time
            print(f'Epoch {current_epoch} 总用时: {epoch_time:.2f}秒, {epoch_time/60:.2f}分钟')
            
            # 打印当前训练epoch和batchsize信息
            print(f'----- 当前训练状态 -----')
            print(f'当前Epoch: [{current_epoch}/{opt["train"]["n_epoch"]-1}]')
            print(f'训练总轮数: {len(train_loader)}')
            print(f'验证总轮数: {len(val_loader)}')
            print(f'学习率: {optimer.param_groups[0]["lr"]:.7f}')
            print(f'--------------------------------')
            
            # 保存训练评估指标，用于后面打印
            train_scores = scores.copy()

            metric.clear()


            ##################
            ### validation ###
            ##################
            cd_model.eval()
            with torch.no_grad():
                if current_epoch % opt['train']['val_freq'] == 0:
                    val_result_path = '{}/val/{}'.format(opt['path_cd']['result'], current_epoch)
                    os.makedirs(val_result_path, exist_ok=True)
                    
                    # 验证阶段记录开始时间
                    val_batch_start_time = time.time()

                    for current_step, val_data in enumerate(val_loader):
                        val_img1 = val_data['A'].to(device)
                        val_img2 = val_data['B'].to(device)
                        pred_img = cd_model(val_img1, val_img2)
                        gt = val_data['L'].to(device).long()
                        val_loss = loss_fun(pred_img, gt)
                        log_dict['loss'] = val_loss.item()
                        #pred score
                        G_pred = pred_img.detach()
                        G_pred = torch.argmax(G_pred, dim=1)
                        current_score = metric.update_cm(pr=G_pred.cpu().numpy(), gt=gt.detach().cpu().numpy())
                        log_dict['running_acc'] = current_score.item()
                        
                        # 每100个batch打印当前批次、用时和学习率
                        if (current_step + 1) % 100 == 0:
                            val_batch_time = time.time() - val_batch_start_time
                            print(f'验证批次: [{current_step + 1}/{len(val_loader)}], 用时: {val_batch_time:.2f}秒, 学习率: {optimer.param_groups[0]["lr"]:.7f}')
                            val_batch_start_time = time.time()

                        # log running batch status for val data
                        if current_step % opt['train']['val_print_iter'] == 0:
                            # message
                            logs = log_dict
                            message = '[Validation CD]. epoch: [%d/%d]. Itter: [%d/%d], running_mf1: %.5f\n' % \
                                      (current_epoch, opt['train']['n_epoch'] - 1, current_step, len(val_loader), logs['running_acc'])
                            logger.info(message)

                            #visual
                            out_dict = OrderedDict()
                            out_dict['pred_cm'] = torch.argmax(pred_img, dim=1, keepdim=False)
                            out_dict['gt_cm'] = gt
                            visuals = out_dict

                            img_mode = "grid"
                            if img_mode == "single":
                                # Converting to uint8
                                img_A = Metrics.tensor2img(val_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                                img_B = Metrics.tensor2img(val_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                                gt_cm = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1),
                                                           out_type=np.uint8, min_max=(0, 1))  # uint8
                                pred_cm = Metrics.tensor2img(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1),
                                                             out_type=np.uint8, min_max=(0, 1))  # uint8

                                # save imgs
                                Metrics.save_img(
                                    img_A, '{}/img_A_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                                Metrics.save_img(
                                    img_B, '{}/img_B_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                                Metrics.save_img(
                                    pred_cm, '{}/img_pred_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                                Metrics.save_img(
                                    gt_cm, '{}/img_gt_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                            else:
                                # grid img
                                visuals['pred_cm'] = visuals['pred_cm'] * 2.0 - 1.0
                                visuals['gt_cm'] = visuals['gt_cm'] * 2.0 - 1.0
                                grid_img = torch.cat((val_data['A'].to(device),
                                                      val_data['B'].to(device),
                                                      visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1),
                                                      visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                                     dim=0)
                                grid_img = Metrics.tensor2img(grid_img)  # uint8
                                Metrics.save_img(
                                    grid_img,'{}/img_A_B_pred_gt_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))

                    ### log epoch status ###
                    scores = metric.get_scores()
                    epoch_acc = scores['mf1']
                    log_dict['epoch_acc'] = epoch_acc.item()
                    for k, v in scores.items():
                        log_dict[k] = v
                    logs = log_dict
                    message = '[Validation CD (epoch summary)]: epoch: [%d/%d]. epoch_mF1=%.5f \n' % \
                              (current_epoch, opt['train']['n_epoch'], logs['epoch_acc'])
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                    message += '\n'
                    logger.info(message)
                    
                    # 保存验证评估指标
                    val_scores = scores.copy()
                    
                    # 打印训练和验证评估指标
                    print('\n')
                    print('-' * 50)
                    print('----- 训练结果 -----')
                    print(f'mF1: {train_scores["mf1"]:.4f} mIoU: {train_scores["miou"]:.4f} 准确率: {train_scores["acc"]:.4f}')
                    print(f'类别0 - F1: {train_scores["F1_0"]:.4f} IoU: {train_scores["iou_0"]:.4f} 精确率: {train_scores["precision_0"]:.4f} 召回率: {train_scores["recall_0"]:.4f}')
                    print(f'类别1 - F1: {train_scores["F1_1"]:.4f} IoU: {train_scores["iou_1"]:.4f} 精确率: {train_scores["precision_1"]:.4f} 召回率: {train_scores["recall_1"]:.4f}')
                    
                    print('\n----- 验证结果 -----')
                    print(f'mF1: {val_scores["mf1"]:.4f} mIoU: {val_scores["miou"]:.4f} 准确率: {val_scores["acc"]:.4f}')
                    print(f'类别0 - F1: {val_scores["F1_0"]:.4f} IoU: {val_scores["iou_0"]:.4f} 精确率: {val_scores["precision_0"]:.4f} 召回率: {val_scores["recall_0"]:.4f}')
                    print(f'类别1 - F1: {val_scores["F1_1"]:.4f} IoU: {val_scores["iou_1"]:.4f} 精确率: {val_scores["precision_1"]:.4f} 召回率: {val_scores["recall_1"]:.4f}')
                    print('-' * 50)
                    print('\n')

                    #best model
                    if logs['epoch_acc'] > best_mF1:
                        is_best_model = True
                        previous_best = best_mF1
                        best_mF1 = logs['epoch_acc']
                        logger.info('[Validation CD] Best model updated. Saving the models (current + best) and training states.')
                        
                        # 打印最佳模型更新信息
                        print('\n')
                        print('*' * 60)
                        print(f'【最佳模型已更新!】')
                        print(f'当前Epoch: {current_epoch}/{opt["train"]["n_epoch"]-1}')
                        print(f'旧的最佳mF1: {previous_best:.6f} -> 新的最佳mF1: {best_mF1:.6f}')
                        if previous_best > 0:
                            improvement_percentage = (best_mF1 - previous_best) / previous_best * 100
                            print(f'验证集性能提升: {best_mF1 - previous_best:.6f} (+{improvement_percentage:.2f}%)')
                        else:
                            print(f'验证集性能提升: {best_mF1 - previous_best:.6f} (首次评估)')
                        print(f'当前验证集详细指标:')
                        print(f'- mF1: {val_scores["mf1"]:.6f}, mIoU: {val_scores["miou"]:.6f}, 准确率: {val_scores["acc"]:.6f}')
                        print(f'- 类别1(变化区域) F1: {val_scores["F1_1"]:.6f}, IoU: {val_scores["iou_1"]:.6f}')
                        print(f'- 精确率: {val_scores["precision_1"]:.6f}, 召回率: {val_scores["recall_1"]:.6f}')
                        print('*' * 60)
                        print('\n')
                        
                        # save model
                        save_network(opt, current_epoch, cd_model, optimer, is_best_model, logs['epoch_acc'])
                    else:
                        is_best_model = False
                        logger.info('[Validation CD]Saving the current cd model and training states.')
                        
                        # 打印模型未更新信息
                        print('\n')
                        print('-' * 50)
                        print(f'【模型未更新】当前mF1({logs["epoch_acc"]:.6f}) <= 最佳mF1({best_mF1:.6f})')
                        print(f'当前与最佳差距: {best_mF1 - logs["epoch_acc"]:.6f} ({(best_mF1 - logs["epoch_acc"]) / best_mF1 * 100:.2f}%)')
                        print('-' * 50)
                        print('\n')
                        
                        # 保存当前模型，但不标记为最佳
                        save_network(opt, current_epoch, cd_model, optimer, is_best_model)
                    logger.info('--- Proceed To The Next Epoch ----\n \n')


                    metric.clear()

            get_scheduler(optimizer=optimer, args=opt['train']).step()
        logger.info('End of training.')

    else:
        logger.info('Begin Model Evaluation (testing).')
        test_result_path = '{}/test/'.format(opt['path_cd']['result'])
        os.makedirs(test_result_path, exist_ok=True)
        logger_test = logging.getLogger('test')  # test logger

        ##load network
        load_path = opt["path_cd"]["resume_state"]
        print(load_path)
        if load_path is not None:
            logger.info(
                'Loading pretrained model for CD model [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)

            # change detection model
            cd_model = Model.create_CD_model(opt)
            cd_model.load_state_dict(torch.load(gen_path), strict=True)
            cd_model.to(device)
            metric.clear()
            cd_model.eval()
            with torch.no_grad():
                for current_step, test_data in enumerate(test_loader):
                    test_img1 = test_data['A'].to(device)
                    test_img2 = test_data['B'].to(device)
                    pred_img = cd_model(test_img1, test_img2)
                    gt = test_data['L'].to(device).long()

                    # pred score
                    G_pred = pred_img.detach()
                    G_pred = torch.argmax(G_pred, dim=1)
                    current_score = metric.update_cm(pr=G_pred.cpu().numpy(), gt=gt.detach().cpu().numpy())
                    log_dict['running_acc'] = current_score.item()

                    logs = log_dict
                    message = '[Testing CD]. Itter: [%d/%d], running_mf1: %.5f\n' % \
                              (current_step, len(test_loader), logs['running_acc'])
                    logger_test.info(message)

                    # Vissuals
                    out_dict = OrderedDict()
                    out_dict['pred_cm'] = torch.argmax(pred_img, dim=1, keepdim=False)
                    out_dict['gt_cm'] = gt
                    visuals = out_dict

                    img_mode = 'single'
                    if img_mode == 'single':
                        # Converting to uint8
                        visuals['pred_cm'] = visuals['pred_cm'] * 2.0 - 1.0
                        visuals['gt_cm'] = visuals['gt_cm'] * 2.0 - 1.0
                        img_A = Metrics.tensor2img(test_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        img_B = Metrics.tensor2img(test_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        gt_cm = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8,
                                                   min_max=(0, 1))  # uint8
                        pred_cm = Metrics.tensor2img(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1),
                                                     out_type=np.uint8, min_max=(0, 1))  # uint8

                        # Save imgs
                        Metrics.save_img(
                            img_A, '{}/img_A_{}.png'.format(test_result_path, current_step))
                        Metrics.save_img(
                            img_B, '{}/img_B_{}.png'.format(test_result_path, current_step))
                        Metrics.save_img(
                            pred_cm, '{}/img_pred_cm{}.png'.format(test_result_path, current_step))
                        Metrics.save_img(
                            gt_cm, '{}/img_gt_cm{}.png'.format(test_result_path, current_step))
                    else:
                        # grid img
                        visuals['pred_cm'] = visuals['pred_cm'] * 2.0 - 1.0
                        visuals['gt_cm'] = visuals['gt_cm'] * 2.0 - 1.0
                        grid_img = torch.cat((test_data['A'],
                                              test_data['B'],
                                              visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1),
                                              visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                             dim=0)
                        grid_img = Metrics.tensor2img(grid_img)  # uint8
                        Metrics.save_img(
                            grid_img, '{}/img_A_B_pred_gt_{}.png'.format(test_result_path, current_step))

                ### log epoch status ###
                scores = metric.get_scores()
                epoch_acc = scores['mf1']
                log_dict['epoch_acc'] = epoch_acc.item()
                for k, v in scores.items():
                    log_dict[k] = v
                logs = log_dict
                message = '[Test CD summary]: Test mF1=%.5f \n' % \
                          (logs['epoch_acc'])
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    message += '\n'
                logger_test.info(message)
                logger.info('End of testing...')
