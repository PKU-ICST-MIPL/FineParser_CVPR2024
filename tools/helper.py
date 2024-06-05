import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch
import torch.nn as nn
import time
import numpy as np
from utils.misc import segment_iou, cal_tiou, seg_pool_1d, seg_pool_3d
from einops import rearrange
import segmentation_models_pytorch as smp
import wandb

from datetime import datetime
settigns_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

end = time.time()

def network_forward_train(base_model, psnet_model, decoder, regressor_delta, 
                          dim_reducer1, dim_reducer2, pred_scores,
                          video_1, label_1_score, video_2, label_2_score, video_1_mask, video_2_mask, mse, optimizer, opti_flag,
                          epoch, batch_idx, batch_num, args, label_1_tas, label_2_tas, bce,
                          pred_tious_5, pred_tious_75, dim_reducer3, video_encoder, segmenter, segment_metrics, 
                          rank, train_len, focal_loss, scheduler, n_iter_per_epoch):
    global end
    
    start = time.time()

    data_time = start - end
    
    ############# Segmentation #############
    
    total_video = torch.cat((video_1, video_2), 0) 
    start_idx = list(range(0, 90, 10))
    video_pack = torch.cat([total_video[:, :, i: i + 16] for i in start_idx])  
    mask_feamap, mask_feature, mask_pred = segmenter(video_pack)  

    Nt, C, T, H, W = mask_feamap.size()
    mask_feature = mask_feature.reshape(len(start_idx), len(total_video), -1).transpose(0, 1)  

    total_mask = torch.cat((video_1_mask, video_2_mask))  
    mask_target = torch.cat([total_mask[:, :, i: i + 16] for i in start_idx])  
    mask_pred = [rearrange(pred, "b c t h w -> (b t) c h w") for pred in mask_pred]  
    mask_target = rearrange(mask_target, "b c t h w -> (b t) c h w")  
    mask_target = mask_target.round().long()

    loss_mask = 0.0
    for i in range(5):
        loss_mask += focal_loss(mask_pred[i], mask_target)

    tp, fp, fn, tn = smp.metrics.get_stats(mask_pred[-1], mask_target, mode='binary', threshold=0.5)
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
    
    segment_metrics["iou_scores"].append(iou_score.item())
    segment_metrics["f1_scores"].append(f1_score.item())
    segment_metrics["f2_scores"].append(f2_score.item())
    segment_metrics["accuracy"].append(accuracy.item())
    segment_metrics["recall"].append(recall.item())
    ############# Video featrue #############
    com_feature_12, com_feamap_12 = base_model(video_1, video_2)
    video_1_fea = com_feature_12[:,:,:com_feature_12.shape[2] // 2]
    video_2_fea = com_feature_12[:,:,com_feature_12.shape[2] // 2:]
    com_feature_12_u = torch.cat((video_1_fea, video_2_fea), 0)
    
    ############# Mask and I3D Feature Fusion #############
    # add
    u_fea_96 = com_feature_12_u*torch.sigmoid(mask_feature)
    video_1_fea = u_fea_96[:u_fea_96.shape[0]//2]
    video_2_fea = u_fea_96[u_fea_96.shape[0]//2:]
    
    ############# Predict transit #############
    y1 = psnet_model(video_1_fea)
    y2 = psnet_model(video_2_fea)
    transits_pred = torch.cat((y1, y2))
    label_12_tas = torch.cat((label_1_tas, label_2_tas), 0)

    label_12_pad = torch.zeros(transits_pred.size())
    for bs in range(transits_pred.shape[0]):
        label_12_pad[bs, int(label_12_tas[bs, 0]), 0] = 1
        label_12_pad[bs, int(label_12_tas[bs, -1]), -1] = 1
    loss_tas = bce(transits_pred, label_12_pad.cuda())

    num = round(transits_pred.shape[1] / transits_pred.shape[-1])
    transits_st_ed = torch.zeros(label_12_tas.size())
    for bs in range(transits_pred.shape[0]):
        for i in range(transits_pred.shape[-1]):
            transits_st_ed[bs, i] = transits_pred[bs, i * num: (i + 1) * num, i].argmax(0).cpu().item() + i * num
    label_1_tas_pred = transits_st_ed[:transits_st_ed.shape[0] // 2]
    label_2_tas_pred = transits_st_ed[transits_st_ed.shape[0] // 2:]
    
    if epoch / args.max_epoch <= args.prob_tas_threshold:
        label_1_tas_pred = label_1_tas
        label_2_tas_pred = label_2_tas
        
    ############# Static feature #############
    v11, v12, v13 = video_encoder(video_1, label_1_tas_pred)
    v21, v22, v23 = video_encoder(video_2, label_2_tas_pred)
    
    ############# Interpolate #############
    N, T, C = video_1_fea.size()
    video_1_fea = video_1_fea.transpose(1, 2)
    video_1_fea_re_list = []
    for bs in range(N):
        v1i0 = int(label_1_tas[bs][0].item() + label_2_tas[bs][0].item()) // 20
        v1i1 = int(label_1_tas[bs][1].item() + label_2_tas[bs][1].item()) // 20
        video_1_fea_re_list.append(seg_pool_1d(video_1_fea[bs].unsqueeze(0), v1i0, v1i1, 4))
    video_1_fea_re = torch.cat(video_1_fea_re_list, 0).transpose(1, 2)

    video_2_fea = video_2_fea.transpose(1, 2)
    video_2_fea_re_list = []
    for bs in range(N):
        v1i0 = int(label_1_tas[bs][0].item() + label_2_tas[bs][0].item()) // 20
        v1i1 = int(label_1_tas[bs][1].item() + label_2_tas[bs][1].item()) // 20
        video_2_fea_re_list.append(seg_pool_1d(video_2_fea[bs].unsqueeze(0), v1i0, v1i1, 4))
    video_2_fea_re = torch.cat(video_2_fea_re_list, 0).transpose(1, 2)

    ############# Lower dimension #############
    video_1_segs_1 = dim_reducer3(video_1_fea_re[:, 0:4, :])
    video_1_segs_2 = dim_reducer1(video_1_fea_re[:, 4:8, :])
    video_1_segs_3 = dim_reducer2(video_1_fea_re[:, 8:12, :])
    video_2_segs_1 = dim_reducer3(video_2_fea_re[:, 0:4, :])
    video_2_segs_2 = dim_reducer1(video_2_fea_re[:, 4:8, :])
    video_2_segs_3 = dim_reducer2(video_2_fea_re[:, 8:12, :])
    
    ############# Fusion dynamic and  static feature #############
    v_12_list = []
    v_21_list = []
    v_12 = decoder[0](v11, v21)
    v_21 = decoder[0](v21, v11)
    v_12_list.append(v_12)
    v_21_list.append(v_21)

    v_12 = decoder[0](v12, v22)
    v_21 = decoder[0](v22, v12)
    v_12_list.append(v_12)
    v_21_list.append(v_21)

    v_12 = decoder[2](v13, v23)
    v_21 = decoder[2](v23, v13)
    v_12_list.append(v_12)
    v_21_list.append(v_21)

    v_12_map = torch.cat(v_12_list, 1)
    v_21_map = torch.cat(v_21_list, 1)
    v_12_21 = torch.cat((v_12_map, v_21_map), 0)
    
    ############# Cross attention #############
    decoder_video_12_map_list = []
    decoder_video_21_map_list = []

    decoder_video_12_map = decoder[1](video_1_segs_1,
                                     video_2_segs_1)
    decoder_video_21_map = decoder[1](video_2_segs_1,
                                     video_1_segs_1)  
    decoder_video_12_map_list.append(decoder_video_12_map)
    decoder_video_21_map_list.append(decoder_video_21_map)

    decoder_video_12_map = decoder[1](video_1_segs_2,
                                     video_2_segs_2) 
    decoder_video_21_map = decoder[1](video_2_segs_2,
                                     video_1_segs_2)  
    decoder_video_12_map_list.append(decoder_video_12_map)
    decoder_video_21_map_list.append(decoder_video_21_map)
    decoder_video_12_map = decoder[3](video_1_segs_3,
                                     video_2_segs_3) 
    decoder_video_21_map = decoder[3](video_2_segs_3,
                                     video_1_segs_3) 
    decoder_video_12_map_list.append(decoder_video_12_map)
    decoder_video_21_map_list.append(decoder_video_21_map)

    decoder_video_12_map = torch.cat(decoder_video_12_map_list, 1)
    decoder_video_21_map = torch.cat(decoder_video_21_map_list, 1)
   
    ############# Fine-grained Contrastive Regression #############
    decoder_12_21 = torch.cat((decoder_video_12_map, decoder_video_21_map), 0)

    delta1 = regressor_delta[0](decoder_12_21)
    delta2 = regressor_delta[1](v_12_21)
    delta1_1 = delta1[:,:12].mean(1)
    delta1_2 = delta1[:,12:24].mean(1)
    delta1_3 = delta1[:,24:].mean(1)
    
    delta2_1 = delta2[:,:4].mean(1)
    delta2_2 = delta2[:,4:8].mean(1)
    delta2_3 = delta2[:,8:].mean(1)
    delta1 = (delta1_1*3+delta1_2*5+delta1_3*2)/10
    delta2 = (delta2_1*3+delta2_2*5+delta2_3*2)/10
    delta=torch.cat((delta1,delta2),1)
    delta = delta.mean(1).unsqueeze(-1) 

    loss_aqa = mse(delta[:delta.shape[0] // 2], (label_1_score - label_2_score)) \
        + mse(delta[delta.shape[0] // 2:], (label_2_score - label_1_score))

    loss = loss_aqa + loss_tas + loss_mask
    loss.backward()
    if (batch_idx+1)%args.update_iter==0 or (batch_idx+1)==train_len:
        optimizer.step()
        optimizer.zero_grad()

    if scheduler:
        scheduler.step_update(epoch * n_iter_per_epoch + batch_idx)

    end = time.time()
    batch_time = end - start
    score = (delta[:delta.shape[0] // 2].detach() + label_2_score)  
    pred_scores.extend([i.item() for i in score])

    tIoU_results = []
    for bs in range(transits_pred.shape[0] // 2):
        tIoU_results.append(segment_iou(np.array(label_12_tas.squeeze(-1).cpu())[bs],
                                        np.array(transits_st_ed.squeeze(-1).cpu())[bs],
                                        args))

    tiou_thresholds = np.array([0.5, 0.75])
    tIoU_correct_per_thr = cal_tiou(tIoU_results, tiou_thresholds)
    Batch_tIoU_5 = tIoU_correct_per_thr[0]
    Batch_tIoU_75 = tIoU_correct_per_thr[1]
    pred_tious_5.extend([Batch_tIoU_5])
    pred_tious_75.extend([Batch_tIoU_75])
    
    if rank == 0:
        if args.wandb and batch_idx % args.log_freq == 0:
            wandb.log({
            'training/Batch_time': batch_time,
            'training/Data_time': data_time,
            'training/Loss': loss.item(),
            'training/Loss_aqa': loss_aqa.item(),
            'training/Loss_tas': loss_tas.item(),
            'training/Loss_mask': loss_mask.item(),
            'training/LR1': optimizer.param_groups[0]['lr'],
            'training/LR2': optimizer.param_groups[1]['lr']
        })
            
        if batch_idx % args.print_freq == 0:
            print('[Training][%d/%d][%d/%d] \t Batch_time: %.2f\t Data_time: %.2f\t Loss: %.4f\t Loss_aqa: %.4f \t Loss_tas: %.4f\t Loss_mask: %.4f\t'
                'lr1 : %0.5f \t lr2 : %0.5f'
                % (epoch, args.max_epoch, batch_idx, batch_num, batch_time, data_time, loss.item(), loss_aqa.item(), loss_tas.item(), loss_mask.item() ,
                    optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
    return loss_aqa.item(), loss_tas.item(), loss_mask.item()


def network_forward_test(base_model, psnet_model, decoder, regressor_delta, video_encoder, dim_reducer3, segmenter, 
                         dim_reducer1, dim_reducer2, pred_scores,
                         video_1, video_2_list, label_2_score_list, video_1_mask, video_2_mask_list,
                         args, label_1_tas, label_2_tas_list,
                         pred_tious_test_5, pred_tious_test_75, segment_metrics,
                         mse, bce, focal_loss, label_1_score):
    score = 0
    tIoU_results = []
    t_loss = [0.0,0.0,0.0]
    for video_2, video_2_mask, label_2_score, label_2_tas in zip(video_2_list, video_2_mask_list, label_2_score_list, label_2_tas_list):
        
        ############# Segmentation #############
    
        total_video = torch.cat((video_1, video_2), 0) 
        start_idx = list(range(0, 90, 10))
        video_pack = torch.cat([total_video[:, :, i: i + 16] for i in start_idx])
        mask_feamap, mask_feature, mask_pred = segmenter(video_pack)  

        Nt, C, T, H, W = mask_feamap.size()
        mask_feature = mask_feature.reshape(len(start_idx), len(total_video), -1).transpose(0, 1)  

        total_mask = torch.cat((video_1_mask, video_2_mask))  
        mask_target = torch.cat([total_mask[:, :, i: i + 16] for i in start_idx])  
        mask_pred_ = [rearrange(pred, "b c t h w -> (b t) c h w") for pred in mask_pred]
        mask_target = rearrange(mask_target, "b c t h w -> (b t) c h w")
        mask_target = mask_target.round().long()
        
        loss_mask = 0.0
        for i in range(5):
            loss_mask += focal_loss(mask_pred_[i], mask_target)
            
        mask_pred = rearrange(mask_pred[-1], "b c t h w -> (b t) c h w")
        tp, fp, fn, tn = smp.metrics.get_stats(mask_pred, mask_target, mode='binary', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        
        segment_metrics["iou_scores"].append(iou_score.item())
        segment_metrics["f1_scores"].append(f1_score.item())
        segment_metrics["f2_scores"].append(f2_score.item())
        segment_metrics["accuracy"].append(accuracy.item())
        segment_metrics["recall"].append(recall.item())
        
        ############# Video featrue #############
        com_feature_12, com_feamap_12 = base_model(video_1, video_2)
        video_1_fea = com_feature_12[:,:,:com_feature_12.shape[2] // 2]
        video_2_fea = com_feature_12[:,:,com_feature_12.shape[2] // 2:]
        com_feature_12_u = torch.cat((video_1_fea, video_2_fea), 0)

        ############# Mask and I3D Feature Fusion #############
        u_fea_96 = com_feature_12_u*torch.sigmoid(mask_feature)
        video_1_fea = u_fea_96[:u_fea_96.shape[0]//2]
        video_2_fea = u_fea_96[u_fea_96.shape[0]//2:]
        
        ############# Predict transit #############
        y1 = psnet_model(video_1_fea)
        y2 = psnet_model(video_2_fea)
        transits_pred = torch.cat((y1, y2))
        label_12_tas = torch.cat((label_1_tas, label_2_tas), 0)
        
        label_12_pad = torch.zeros(transits_pred.size())
        for bs in range(transits_pred.shape[0]):
            label_12_pad[bs, int(label_12_tas[bs, 0]), 0] = 1
            label_12_pad[bs, int(label_12_tas[bs, -1]), -1] = 1
        loss_tas = bce(transits_pred, label_12_pad.cuda())


        num = round(transits_pred.shape[1] / transits_pred.shape[-1])
        transits_st_ed = torch.zeros(label_12_tas.size())
        for bs in range(transits_pred.shape[0]):
            for i in range(transits_pred.shape[-1]):
                transits_st_ed[bs, i] = transits_pred[bs, i * num: (i + 1) * num, i].argmax(0).cpu().item() + i * num
        label_1_tas_pred = transits_st_ed[:transits_st_ed.shape[0] // 2]
        label_2_tas_pred = transits_st_ed[transits_st_ed.shape[0] // 2:]
            
        ############# Static feature #############
        v11, v12, v13 = video_encoder(video_1, label_1_tas_pred)
        v21, v22, v23 = video_encoder(video_2, label_2_tas_pred)
        
        ############# Interpolate #############
        N, T, C = video_1_fea.size()
        video_1_fea = video_1_fea.transpose(1, 2)
        video_1_fea_re_list = []
        for bs in range(N):
            v1i0 = int(label_1_tas_pred[bs][0].item() + label_2_tas_pred[bs][0].item()) // 20
            v1i1 = int(label_1_tas_pred[bs][1].item() + label_2_tas_pred[bs][1].item()) // 20
            video_1_fea_re_list.append(seg_pool_1d(video_1_fea[bs].unsqueeze(0), v1i0, v1i1, 4))
        video_1_fea_re = torch.cat(video_1_fea_re_list, 0).transpose(1, 2)

        video_2_fea = video_2_fea.transpose(1, 2)
        video_2_fea_re_list = []
        for bs in range(N):
            v1i0 = int(label_1_tas_pred[bs][0].item() + label_2_tas_pred[bs][0].item()) // 20
            v1i1 = int(label_1_tas_pred[bs][1].item() + label_2_tas_pred[bs][1].item()) // 20
            video_2_fea_re_list.append(seg_pool_1d(video_2_fea[bs].unsqueeze(0), v1i0, v1i1, 4))
        video_2_fea_re = torch.cat(video_2_fea_re_list, 0).transpose(1, 2)

        ############# Lower dimension #############
        video_1_segs_1 = dim_reducer3(video_1_fea_re[:, 0:4, :])
        video_1_segs_2 = dim_reducer1(video_1_fea_re[:, 4:8, :])
        video_1_segs_3 = dim_reducer2(video_1_fea_re[:, 8:12, :])
        video_2_segs_1 = dim_reducer3(video_2_fea_re[:, 0:4, :])
        video_2_segs_2 = dim_reducer1(video_2_fea_re[:, 4:8, :])
        video_2_segs_3 = dim_reducer2(video_2_fea_re[:, 8:12, :])
        
        ############# Fusion dynamic and  static feature #############
        v_12_list = []
        v_21_list = []
        v_12 = decoder[0](v11, v21)
        v_21 = decoder[0](v21, v11)
        v_12_list.append(v_12)
        v_21_list.append(v_21)

        v_12 = decoder[0](v12, v22)
        v_21 = decoder[0](v22, v12)
        v_12_list.append(v_12)
        v_21_list.append(v_21)

        v_12 = decoder[2](v13, v23)
        v_21 = decoder[2](v23, v13)
        v_12_list.append(v_12)
        v_21_list.append(v_21)

        v_12_map = torch.cat(v_12_list, 1)
        v_21_map = torch.cat(v_21_list, 1)
        v_12_21 = torch.cat((v_12_map, v_21_map), 0)
        
        ############# Cross attention #############
        decoder_video_12_map_list = []
        decoder_video_21_map_list = []

        decoder_video_12_map = decoder[1](video_1_segs_1,
                                        video_2_segs_1)
        decoder_video_21_map = decoder[1](video_2_segs_1,
                                        video_1_segs_1)  
        decoder_video_12_map_list.append(decoder_video_12_map)
        decoder_video_21_map_list.append(decoder_video_21_map)

        decoder_video_12_map = decoder[1](video_1_segs_2,
                                        video_2_segs_2)  
        decoder_video_21_map = decoder[1](video_2_segs_2,
                                        video_1_segs_2)  
        decoder_video_12_map_list.append(decoder_video_12_map)
        decoder_video_21_map_list.append(decoder_video_21_map)
        decoder_video_12_map = decoder[3](video_1_segs_3,
                                        video_2_segs_3)  
        decoder_video_21_map = decoder[3](video_2_segs_3,
                                        video_1_segs_3)  
        decoder_video_12_map_list.append(decoder_video_12_map)
        decoder_video_21_map_list.append(decoder_video_21_map)

        decoder_video_12_map = torch.cat(decoder_video_12_map_list, 1)
        decoder_video_21_map = torch.cat(decoder_video_21_map_list, 1)
        

        ############# Fine-grained Contrastive Regression #############
        decoder_12_21 = torch.cat((decoder_video_12_map, decoder_video_21_map), 0)

        delta1 = regressor_delta[0](decoder_12_21)
        delta2 = regressor_delta[1](v_12_21)
        
        delta1_1 = delta1[:,:12].mean(1)
        delta1_2 = delta1[:,12:24].mean(1)
        delta1_3 = delta1[:,24:].mean(1)
        
        delta2_1 = delta2[:,:4].mean(1)
        delta2_2 = delta2[:,4:8].mean(1)
        delta2_3 = delta2[:,8:].mean(1)
        delta1 = (delta1_1*3+delta1_2*5+delta1_3*2)/10
        delta2 = (delta2_1*3+delta2_2*5+delta2_3*2)/10
        delta=torch.cat((delta1,delta2),1)
        delta = delta.mean(1).unsqueeze(-1) 
        
        score += (delta[:delta.shape[0] // 2].detach() + label_2_score)
        
        loss_aqa = mse(delta[:delta.shape[0] // 2], (label_1_score - label_2_score))
        
        t_loss[0] += loss_aqa
        t_loss[1] += loss_tas
        t_loss[2] += loss_mask
        
        for bs in range(N):
            tIoU_results.append(segment_iou(np.array(label_12_tas.squeeze(-1).cpu())[bs],
                                            np.array(transits_st_ed.squeeze(-1).cpu())[bs], args))
        

    pred_scores.extend([i.item() / len(video_2_list) for i in score])

    tIoU_results_mean = [sum(tIoU_results) / len(tIoU_results)]
    tiou_thresholds = np.array([0.5, 0.75])
    tIoU_correct_per_thr = cal_tiou(tIoU_results_mean, tiou_thresholds)
    pred_tious_test_5.extend([tIoU_correct_per_thr[0]])
    pred_tious_test_75.extend([tIoU_correct_per_thr[1]])
    
    for i in range(len(t_loss)):
        t_loss[i] /= args.voter_number
    return t_loss


def save_checkpoint(base_model, psnet_model, decoder, regressor_delta, video_encoder, dim_reducer3, segmenter,
                    dim_reducer1,dim_reducer2,
                    optimizer, epoch, epoch_best_aqa, rho_best, L2_min, RL2_min, prefix, args):
    torch.save({
        'base_model': base_model.state_dict(),
        'psnet_model': psnet_model.state_dict(),
        'decoder1': decoder[0].state_dict(),
        'decoder2': decoder[1].state_dict(),
        'decoder3': decoder[2].state_dict(),
        'decoder4': decoder[3].state_dict(),
        'regressor_delta1': regressor_delta[0].state_dict(),
        'regressor_delta2': regressor_delta[1].state_dict(),
        'regressor_delta3': regressor_delta[2].state_dict(),
        'video_encoder': video_encoder.state_dict(),
        'dim_reducer3': dim_reducer3.state_dict(),
        'dim_reducer1': dim_reducer1.state_dict(),
        'dim_reducer2': dim_reducer2.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'epoch_best_aqa': epoch_best_aqa,
        'rho_best': rho_best,
        'L2_min': L2_min,
        'RL2_min': RL2_min,
        'segmenter': segmenter.state_dict(),
    }, os.path.join(args.experiment_path, prefix + '.pth'))

def save_outputs(pred_scores, true_scores, args, epoch):
    save_path_pred = os.path.join(args.experiment_path, f'pred_{settigns_date_time}_{epoch}.npy')
    save_path_true = os.path.join(args.experiment_path, f'true_{settigns_date_time}_{epoch}.npy')
    np.save(save_path_pred, pred_scores)
    np.save(save_path_true, true_scores)
