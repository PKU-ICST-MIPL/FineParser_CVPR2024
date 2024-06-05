import numpy as np
import torch
import torch.nn as nn

from scipy import stats
from tools import builder, helper
from utils import misc
import time
import json

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import all_reduce, ReduceOp
import wandb
import segmentation_models_pytorch as smp


def setup(rank, world_size, args):
    dist.init_process_group(backend='nccl',  init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    if rank == 0 and args.wandb:
        wandb.init(project="FineParser")
    

def train_net(args, rank, world_size):
    setup(rank, world_size, args)
    print('Trainer start ... ')
    # build dataset
    train_dataset, test_dataset = builder.dataset_builder(args)
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        num_replicas=world_size, rank=rank)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, 
                                                                    num_replicas=world_size, rank=rank, shuffle=False)
        

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs_train,
                                                    shuffle=False, num_workers=int(args.workers),
                                                    pin_memory=True, sampler=train_sampler,
                                                    worker_init_fn=misc.worker_init_fn)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs_test,
                                                    shuffle=False, num_workers=int(args.workers),
                                                    pin_memory=True, sampler=test_sampler)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs_train,
                                                    shuffle=True , num_workers=int(args.workers),
                                                    pin_memory=True, 
                                                    worker_init_fn=misc.worker_init_fn)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs_test,
                                                    shuffle=False, num_workers=int(args.workers),
                                                    pin_memory=True, )
    
    # build model
    base_model, psnet_model, decoder, regressor_delta, dim_reducer1, \
        dim_reducer2, video_encoder, dim_reducer3, segmenter = builder.model_builder(args)

    # CUDA
    global use_gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_model = base_model.cuda()
        psnet_model = psnet_model.cuda()
        decoder = [m.cuda() for m in decoder]
        regressor_delta = [m.cuda() for m in regressor_delta]
        dim_reducer1 = dim_reducer1.cuda()
        dim_reducer2 = dim_reducer2.cuda()
        video_encoder = video_encoder.cuda()
        segmenter = segmenter.cuda()
        dim_reducer3 = dim_reducer3.cuda()

    n_iter_per_epoch = len(train_dataloader)
    optimizer, scheduler = builder.build_opti_sche(base_model, psnet_model, decoder, regressor_delta,
                                                        dim_reducer1, dim_reducer2, dim_reducer3, video_encoder, segmenter, args, n_iter_per_epoch)

    start_epoch = 0
    global epoch_best_tas, pred_tious_best_5, pred_tious_best_75, epoch_best_aqa, rho_best, L2_min, RL2_min
    epoch_best_tas = 0
    pred_tious_best_5 = 0
    pred_tious_best_75 = 0
    epoch_best_aqa = 0
    rho_best = 0
    L2_min = 1000
    RL2_min = 1000

    global epoch_best_seg, best_iou_score, best_f1_score,best_f2_score, best_accuracy, best_recall
    epoch_best_seg = 0
    best_iou_score = 0
    best_f1_score = 0
    best_f2_score = 0
    best_accuracy = 0
    best_recall = 0

    # resume ckpts
    if args.resume or args.test:
        start_epoch, epoch_best_aqa, rho_best, L2_min, RL2_min = builder.resume_train(base_model, psnet_model, decoder,
                            dim_reducer1, dim_reducer2, optimizer, dim_reducer3, regressor_delta, segmenter,video_encoder, args)
        print('resume ckpts @ %d epoch(rho = %.4f, L2 = %.4f , RL2 = %.4f)'
              % (start_epoch - 1, rho_best, L2_min, RL2_min))

    # DDP
    if world_size > 1:
        base_model = DDP(base_model, device_ids=[rank])
        psnet_model = DDP(psnet_model, device_ids=[rank])
        decoder = [DDP(_decoder, device_ids=[rank]) for _decoder in decoder]
        regressor_delta = [DDP(_regressor_delta, device_ids=[rank]) for _regressor_delta in regressor_delta]
        dim_reducer1 = DDP(dim_reducer1)
        dim_reducer2 = DDP(dim_reducer2)
        dim_reducer3 = DDP(dim_reducer3)
        segmenter = DDP(segmenter, device_ids=[rank])
        video_encoder = DDP(video_encoder, device_ids=[rank], find_unused_parameters=True)
        
        base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
        psnet_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(psnet_model)
        decoder = [torch.nn.SyncBatchNorm.convert_sync_batchnorm(m) for m in decoder]
        regressor_delta = [torch.nn.SyncBatchNorm.convert_sync_batchnorm(m) for m in regressor_delta]
        dim_reducer1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dim_reducer1)
        dim_reducer2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dim_reducer2)
        dim_reducer3 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dim_reducer3)
        segmenter = torch.nn.SyncBatchNorm.convert_sync_batchnorm(segmenter)
        video_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(video_encoder)
    
    # loss
    mse = nn.MSELoss().cuda()
    bce = nn.BCELoss().cuda()
    focal_loss = smp.losses.FocalLoss('binary').cuda()

    if args.test:
        torch.distributed.barrier()
        validate(base_model, psnet_model, decoder, regressor_delta, video_encoder, dim_reducer3, segmenter,
                     dim_reducer1, dim_reducer2, 
                        test_dataloader, -1, optimizer, args, rank)
        exit(0)

    # training phase
    for epoch in range(start_epoch, args.max_epoch):
        if world_size > 1:
            torch.distributed.barrier()
            train_sampler.set_epoch(epoch)
        pred_tious_5 = []
        pred_tious_75 = []
        true_scores = []
        pred_scores = []
        
        segment_metrics = {
            "iou_scores": [],
            "f1_scores": [],
            "f2_scores": [],
            "accuracy": [],
            "recall": [],
        }

        base_model.train()  
        psnet_model.train()
        for _decoder in decoder:
            _decoder.train()
        for _regressor_delta in regressor_delta:
            _regressor_delta.train()
        dim_reducer3.train()
        dim_reducer1.train()
        dim_reducer2.train()
        segmenter.train()
        video_encoder.train()

        if args.fix_bn:
            base_model.apply(misc.fix_bn)
        
        total_loss = [0.0, 0.0, 0.0]
        for idx, (data, target) in enumerate(train_dataloader):
            # num_iter += 1
            opti_flag = True

            # video_1 is query and video_2 is exemplar
            video_1 = data['video'].float().cuda()
            video_2 = target['video'].float().cuda()
            video_1_mask = data['video_mask'].float().cuda()
            video_2_mask = target['video_mask'].float().cuda()
            label_1_tas = data['transits'].float().cuda() + 1
            label_2_tas = target['transits'].float().cuda() + 1
            label_1_score = data['final_score'].float().reshape(-1, 1).cuda()
            label_2_score = target['final_score'].float().reshape(-1, 1).cuda()

            # forward
            res = helper.network_forward_train(base_model, psnet_model, decoder, regressor_delta, 
                                         dim_reducer1, dim_reducer2, pred_scores,
                                         video_1, label_1_score, video_2, label_2_score, video_1_mask, video_2_mask, mse, optimizer,
                                         opti_flag, epoch, idx+1, len(train_dataloader),
                                         args, label_1_tas, label_2_tas, bce,
                                         pred_tious_5, pred_tious_75, dim_reducer3, video_encoder, segmenter, segment_metrics, 
                                         rank, len(train_dataloader), focal_loss, scheduler, n_iter_per_epoch)
            
            for t in range(len(res)):
                total_loss[t] += res[t]
            
            true_scores.extend(data['final_score'].numpy())
        
        t_loss = []
        for l in total_loss:
            loss_tensor = torch.tensor(l).cuda()
            dist.all_reduce(loss_tensor, op=ReduceOp.SUM)
            t_loss.append(loss_tensor)
        
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        
        pred_scores_tensor = torch.tensor(pred_scores).cuda()
        true_scores_tensor = torch.tensor(true_scores).cuda()
        
        gathered_pred_scores = [torch.zeros_like(pred_scores_tensor) for _ in range(dist.get_world_size())]
        gathered_true_scores = [torch.zeros_like(true_scores_tensor) for _ in range(dist.get_world_size())]
        
        dist.all_gather(gathered_pred_scores, pred_scores_tensor)
        dist.all_gather(gathered_true_scores, true_scores_tensor)
        
        sum_iou = torch.tensor(sum(segment_metrics["iou_scores"])).cuda()
        sum_f1 = torch.tensor(sum(segment_metrics["f1_scores"])).cuda()
        sum_f2 = torch.tensor(sum(segment_metrics["f2_scores"])).cuda()
        sum_accuracy = torch.tensor(sum(segment_metrics["accuracy"])).cuda()
        sum_recall = torch.tensor(sum(segment_metrics["recall"])).cuda()

        dist.all_reduce(sum_iou, op=ReduceOp.SUM)
        dist.all_reduce(sum_f1, op=ReduceOp.SUM)
        dist.all_reduce(sum_f2, op=ReduceOp.SUM)
        dist.all_reduce(sum_accuracy, op=ReduceOp.SUM)
        dist.all_reduce(sum_recall, op=ReduceOp.SUM)
        
        sum_test_iou_5 = torch.tensor(sum(pred_tious_5)).cuda()
        sum_test_iou_75 = torch.tensor(sum(pred_tious_75)).cuda()
        
        dist.all_reduce(sum_test_iou_5, op=ReduceOp.SUM)
        dist.all_reduce(sum_test_iou_75, op=ReduceOp.SUM)

        if rank == 0:
            gathered_pred_scores = torch.cat(gathered_pred_scores).cpu().numpy()
            gathered_true_scores = torch.cat(gathered_true_scores).cpu().numpy()
            rho, p = stats.spearmanr(gathered_pred_scores, gathered_true_scores)
            L2 = np.power(gathered_pred_scores - gathered_true_scores, 2).sum() / gathered_true_scores.shape[0]
            RL2 = np.power((gathered_pred_scores - gathered_true_scores) / (gathered_true_scores.max() - gathered_true_scores.min()), 2).sum() / gathered_true_scores.shape[0]
            
            pred_tious_mean_5 = (sum_test_iou_5 / (len(train_dataloader) * args.bs_train)) / dist.get_world_size()
            pred_tious_mean_75 = (sum_test_iou_75 / (len(train_dataloader) * args.bs_train)) / dist.get_world_size()
            
            iou_score = (sum_iou.item() / dist.get_world_size()) / (len(train_dataloader) * args.bs_train) 
            f1_score = (sum_f1.item() / dist.get_world_size()) / (len(train_dataloader) * args.bs_train)
            f2_score = (sum_f2.item() / dist.get_world_size()) / (len(train_dataloader) * args.bs_train)
            accuracy = (sum_accuracy.item() / dist.get_world_size()) / (len(train_dataloader) * args.bs_train)
            recall = (sum_recall.item() / dist.get_world_size())/ (len(train_dataloader) * args.bs_train)
            
            print('[Training] EPOCH: %d, tIoU_5: %.4f, tIoU_75: %.4f, Loss_aqa: %.4f, Loss_tas: %.4f, Loss_mask: %.4f'
                % (epoch, pred_tious_mean_5, pred_tious_mean_75, t_loss[0], t_loss[1], t_loss[2]))
            print('[Training] EPOCH: %d, correlation: %.4f, L2: %.4f, RL2: %.4f, lr1: %.4f, lr2: %.4f'%(epoch, rho, L2, RL2, 
                optimizer.param_groups[0]['lr'],  optimizer.param_groups[1]['lr']))
            print('[Training] EPOCH: %d, seg_iou: %.4f, seg_f1: %.4f, seg_f2: %.4f, seg_acc: %.4f, seg_rec: %.4f'%(epoch, iou_score,
                f1_score, f2_score, accuracy, recall))
            
            if args.wandb:
                wandb.log({
                    'train/tIoU_5': pred_tious_mean_5,
                    'train/tIoU_75': pred_tious_mean_75,
                    'train/Epoch_Loss_aqa': t_loss[0],
                    'train/Epoch_Loss_tas': t_loss[1],
                    'train/Epoch_Loss_mask': t_loss[2],
                    'train/correlation': rho,
                    'train/L2': L2,
                    'train/RL2': RL2,
                    'train/seg_iou': iou_score,
                    'train/seg_f1': f1_score,
                    'train/seg_f2': f2_score,
                    'train/seg_acc': accuracy,
                    'train/seg_rec': recall
                })

            
        if epoch < 50 and (epoch+1)%4==0 or epoch>=50 and (epoch+1)%2==0:
            validate(base_model, psnet_model, decoder, regressor_delta, video_encoder, dim_reducer3, segmenter,
                     dim_reducer1, dim_reducer2, 
                        test_dataloader, epoch, optimizer, args, rank)
            if rank == 0:
                print('[TEST] EPOCH: %d, best correlation: %.6f, best L2: %.6f, best RL2: %.6f' % (epoch_best_aqa,
                                                                                            rho_best, L2_min, RL2_min))
                print('[TEST] EPOCH: %d, best tIoU_5: %.6f, best tIoU_75: %.6f' % (epoch_best_tas,
                                                                            pred_tious_best_5, pred_tious_best_75))
                print('[TEST] EPOCH: %d, best seg_iou: %.6f, best seg_f1: %.6f,  best seg_f2: %.6f,  best seg_acc: %.6f,  best seg_rec: %.6f' % 
                    (epoch_best_seg, best_iou_score, best_f1_score, best_f2_score, best_accuracy, best_recall))
        if rank == 0:
            helper.save_checkpoint(base_model, psnet_model, decoder, regressor_delta, video_encoder, dim_reducer3, segmenter, 
                                    dim_reducer1,dim_reducer2,
                                   optimizer, epoch, epoch_best_aqa, rho_best, L2_min, RL2_min, 'last', args)
        

        # scheduler lr
        if scheduler is not None:
            scheduler.step(epoch)


def validate(base_model, psnet_model, decoder, regressor_delta, video_encoder, dim_reducer3, segmenter, 
             dim_reducer1, dim_reducer2, test_dataloader, epoch, optimizer, args, rank):

    if rank == 0:
        print("Start validating epoch {}".format(epoch))
    global use_gpu
    global epoch_best_aqa, rho_best, L2_min, RL2_min, epoch_best_tas, pred_tious_best_5, pred_tious_best_75
    global epoch_best_seg, best_iou_score, best_f1_score, best_f2_score, best_accuracy, best_recall


    true_scores = []
    pred_scores = []
    pred_tious_test_5 = []
    pred_tious_test_75 = []
    
    segment_metrics = {
        "iou_scores": [],
        "f1_scores": [],
        "f2_scores": [],
        "accuracy": [],
        "recall": [],
    }

    base_model.eval()  
    psnet_model.eval()
    for _decoder in decoder:
        _decoder.eval()
    for _regressor_delta in regressor_delta:
        _regressor_delta.eval()
    video_encoder.eval()
    dim_reducer3.eval()
    dim_reducer1.eval()
    dim_reducer2.eval()
    segmenter.eval()
    
    mse = nn.MSELoss().cuda()
    bce = nn.BCELoss().cuda()
    focal_loss = smp.losses.FocalLoss('binary').cuda()
    
    batch_num = len(test_dataloader)
    total_loss = [0.0,0.0,0.0]
    with torch.no_grad():
        datatime_start = time.time()

        for batch_idx, (data, target) in enumerate(test_dataloader, 0):
            datatime = time.time() - datatime_start
            start = time.time()

            video_1 = data['video'].float().cuda()
            video_1_mask = data['video_mask'].float().cuda()
            video_2_mask_list = [item['video_mask'].float().cuda() for item in target]
            video_2_list = [item['video'].float().cuda() for item in target]
            label_1_tas = data['transits'].float().cuda() + 1
            label_2_tas_list = [item['transits'].float().cuda() + 1 for item in target]
            label_2_score_list = [item['final_score'].float().reshape(-1, 1).cuda() for item in target]
            label_1_score = data['final_score'].float().cuda()

            res = helper.network_forward_test(base_model, psnet_model, decoder, regressor_delta, video_encoder, dim_reducer3, segmenter,
                                        dim_reducer1, dim_reducer2, pred_scores,
                                        video_1, video_2_list, label_2_score_list, video_1_mask, video_2_mask_list,
                                        args, label_1_tas, label_2_tas_list, 
                                        pred_tious_test_5, pred_tious_test_75, segment_metrics,
                                        mse, bce, focal_loss, label_1_score)
            
            for t in range(len(res)):
                total_loss[t] += res[t]

            batch_time = time.time() - start
            if rank == 0 and batch_idx % args.print_freq == 0:
                print('[TEST][%d/%d][%d/%d] \t Batch_time %.2f \t Data_time %.2f'
                    % (epoch, args.max_epoch, batch_idx, batch_num, batch_time, datatime))
            datatime_start = time.time()
            true_scores.extend(data['final_score'].numpy())
        
        # evaluation results
        t_loss = []
        for l in total_loss:
            loss_tensor = torch.tensor(l).cuda()
            dist.all_reduce(loss_tensor, op=ReduceOp.SUM)
            t_loss.append(loss_tensor)
        
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        
        pred_scores_tensor = torch.tensor(pred_scores).cuda()
        true_scores_tensor = torch.tensor(true_scores).cuda()
        
        gathered_pred_scores = [torch.zeros_like(pred_scores_tensor) for _ in range(dist.get_world_size())]
        gathered_true_scores = [torch.zeros_like(true_scores_tensor) for _ in range(dist.get_world_size())]
        
        dist.all_gather(gathered_pred_scores, pred_scores_tensor)
        dist.all_gather(gathered_true_scores, true_scores_tensor)
        
        sum_iou = torch.tensor(sum(segment_metrics["iou_scores"])).cuda()
        sum_f1 = torch.tensor(sum(segment_metrics["f1_scores"])).cuda()
        sum_f2 = torch.tensor(sum(segment_metrics["f2_scores"])).cuda()
        sum_accuracy = torch.tensor(sum(segment_metrics["accuracy"])).cuda()
        sum_recall = torch.tensor(sum(segment_metrics["recall"])).cuda()

        # 使用 all_reduce 聚合数据
        dist.all_reduce(sum_iou, op=ReduceOp.SUM)
        dist.all_reduce(sum_f1, op=ReduceOp.SUM)
        dist.all_reduce(sum_f2, op=ReduceOp.SUM)
        dist.all_reduce(sum_accuracy, op=ReduceOp.SUM)
        dist.all_reduce(sum_recall, op=ReduceOp.SUM)
        
        sum_test_iou_5 = torch.tensor(sum(pred_tious_test_5)).cuda()
        sum_test_iou_75 = torch.tensor(sum(pred_tious_test_75)).cuda()
        
        dist.all_reduce(sum_test_iou_5, op=ReduceOp.SUM)
        dist.all_reduce(sum_test_iou_75, op=ReduceOp.SUM)
        
        if rank == 0:
            gathered_pred_scores = torch.cat(gathered_pred_scores).cpu().numpy()
            gathered_true_scores = torch.cat(gathered_true_scores).cpu().numpy()
            rho, p = stats.spearmanr(gathered_pred_scores, gathered_true_scores)
            L2 = np.power(gathered_pred_scores - gathered_true_scores, 2).sum() / gathered_true_scores.shape[0]
            RL2 = np.power((gathered_pred_scores - gathered_true_scores) / (gathered_true_scores.max() - gathered_true_scores.min()), 2).sum() / gathered_true_scores.shape[0]
            pred_tious_test_mean_5 = (sum_test_iou_5 / (len(test_dataloader) * args.bs_test)) / dist.get_world_size()
            pred_tious_test_mean_75 = (sum_test_iou_75 / (len(test_dataloader) * args.bs_test)) / dist.get_world_size()

            if pred_tious_test_mean_5 > pred_tious_best_5:
                pred_tious_best_5 = pred_tious_test_mean_5
            if pred_tious_test_mean_75 > pred_tious_best_75:
                pred_tious_best_75 = pred_tious_test_mean_75
                epoch_best_tas = epoch
        
            current_iou_score = (sum_iou.item() / dist.get_world_size()) / (len(test_dataloader) * args.bs_test) / args.voter_number
            current_f1_score = (sum_f1.item() / dist.get_world_size()) / (len(test_dataloader) * args.bs_test) / args.voter_number
            current_f2_score = (sum_f2.item() / dist.get_world_size()) / (len(test_dataloader) * args.bs_test)/ args.voter_number
            current_accuracy = (sum_accuracy.item() / dist.get_world_size()) / (len(test_dataloader) * args.bs_test)/ args.voter_number
            current_recall = (sum_recall.item() / dist.get_world_size())/ (len(test_dataloader) * args.bs_test)/ args.voter_number
            
            
            if current_iou_score > best_iou_score:
                best_iou_score = current_iou_score
                epoch_best_seg = epoch  

            if current_f1_score > best_f1_score:
                best_f1_score = current_f1_score

            if current_f2_score > best_f2_score:
                best_f2_score = current_f2_score

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy

            if current_recall > best_recall:
                best_recall = current_recall
                
            if L2_min > L2:
                L2_min = L2
            if RL2_min > RL2:
                RL2_min = RL2
            if rho > rho_best:
                rho_best = rho
                epoch_best_aqa = epoch
                print('-----New best found!-----')
                helper.save_checkpoint(base_model, psnet_model, decoder, regressor_delta, video_encoder, dim_reducer3, segmenter, 
                                   dim_reducer1,dim_reducer2,
                                   optimizer, epoch, epoch_best_aqa, rho_best, L2_min, RL2_min, 'best', args)
            helper.save_outputs(gathered_pred_scores, gathered_true_scores, args, epoch)
            print('[TEST] EPOCH: %d, Loss_aqa: %.6f, Loss_tas: %.6f, Loss_mask: %.6f' % (epoch, t_loss[0], t_loss[1], t_loss[2]))
            print('[TEST] EPOCH: %d, tIoU_5: %.6f, tIoU_75: %.6f' % (epoch, pred_tious_best_5, pred_tious_best_75))
            print('[TEST] EPOCH: %d, correlation: %.6f, L2: %.6f, RL2: %.6f' % (epoch, rho, L2, RL2))
            print('[TEST] EPOCH: %d, IOU Score: %.6f, F1 Score: %.6f, F2 Score: %.6f, Accuracy: %.6f, Recall: %.6f' % (epoch, current_iou_score, current_f1_score, current_f2_score, current_accuracy, current_recall))
            
            if args.wandb:
                wandb.log({
                    'test/Loss_aqa': t_loss[0],
                    'test/Loss_tas': t_loss[1],
                    'test/Loss_mask': t_loss[2],
                    'test/tIoU_5': pred_tious_best_5,
                    'test/tIoU_75': pred_tious_best_75,
                    'test/correlation': rho,
                    'test/L2': L2,
                    'test/RL2': RL2,
                    'test/IOU_Score': current_iou_score,
                    'test/F1_Score': current_f1_score,
                    'test/F2_Score': current_f2_score,
                    'test/Accuracy': current_accuracy,
                    'test/Recall': current_recall
                })
