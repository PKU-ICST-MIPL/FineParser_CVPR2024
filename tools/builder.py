import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch
import torch.optim as optim
from models import I3D_backbone
from models.PS import PSNet
from utils.misc import import_class
from torchvideotransforms import video_transforms, volume_transforms
from models import decoder_fuser
from models import MLP_score
from models import decoder_fuser
from models import MLP_score, MLP_twist, I3D_VOS
from models import MaskEncoder, AttentionFusion, FeatureFusionModule, VideoEncoder, C_channel
from models.PS import PSNet, Pred_twistoffset
from datasets.FineDiving_Pair import DebugDataset
import torch
import torchvision.transforms.functional as F
import random
import PIL
from timm.scheduler.cosine_lr import CosineLRScheduler


def get_video_trans():
    train_trans = DualCompose([
        DualRandomHorizontalFlip(),
        DualResize((200, 112)),
        DualRandomCrop(112),
        DualClipToTensor(),
        DualNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_trans = DualCompose([
        DualResize((200, 112)),
        DualCenterCrop(112),
        DualClipToTensor(),
        DualNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_trans, test_trans

def get_video_trans1():
    train_trans = video_transforms.Compose([
        video_transforms.RandomHorizontalFlip(),
        video_transforms.Resize((200, 112)),
        video_transforms.RandomCrop(112),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_trans = video_transforms.Compose([
        video_transforms.Resize((200, 112)),
        video_transforms.CenterCrop(112),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_trans, test_trans

def dataset_builder(args):
    train_trans, test_trans = get_video_trans()
    Dataset = import_class("datasets." + args.benchmark)
    train_dataset = Dataset(args, transform=train_trans, subset='train')
    test_dataset = Dataset(args, transform=test_trans, subset='test')

    return train_dataset, test_dataset


def model_builder(args):
    base_model = I3D_backbone(I3D_class=400)
    base_model.load_pretrain(args.pretrained_i3d_weight)
    PSNet_model = PSNet()
    Decoder_vit1 = decoder_fuser(dim=64, num_heads=8, num_layers=3)
    Decoder_vit2 = decoder_fuser(dim=64, num_heads=8, num_layers=3)
    Decoder_vit3 = decoder_fuser(dim=64, num_heads=8, num_layers=3)
    Decoder_vit4 = decoder_fuser(dim=64, num_heads=8, num_layers=3)
    Regressor_delta1 = MLP_score(in_channel=64, out_channel=1)
    Regressor_delta2 = MLP_score(in_channel=64, out_channel=1)
    Regressor_delta3 = MLP_score(in_channel=64, out_channel=1)
    dim_reducer1  = C_channel(in_channel=4)
    dim_reducer2 = C_channel(in_channel=4)
    dim_reducer3 = C_channel(in_channel=4)
    segmenter = I3D_VOS(num_classes=400)
    Video_Encoder = VideoEncoder()

    return base_model, PSNet_model, [Decoder_vit1, Decoder_vit2, Decoder_vit3,Decoder_vit4], \
        [Regressor_delta1, Regressor_delta2, Regressor_delta3], dim_reducer1, dim_reducer2,\
            Video_Encoder, dim_reducer3, segmenter

def build_opti_sche(base_model, psnet_model, decoder, regressor_delta, dim_reducer1, dim_reducer2, 
                    dim_reducer3, video_encoder, segmenter, args, n_iter_per_epoch):
    if args.optimizer == 'Adam':
        optimizer = optim.Adam([
            {'params': base_model.parameters(), 'lr': args.base_lr * args.lr_factor},
            {'params': psnet_model.parameters()},
            {'params': decoder[0].parameters()},
            {'params': decoder[1].parameters()},
            {'params': decoder[2].parameters()},
            {'params': decoder[3].parameters()},
            {'params': regressor_delta[0].parameters()},
            {'params': regressor_delta[1].parameters()},
            {'params': regressor_delta[2].parameters()},
            {'params': dim_reducer3.parameters()},
            {'params': segmenter.parameters()},
            {'params': video_encoder.parameters()},
            {'params': dim_reducer1.parameters()},
            {'params': dim_reducer2.parameters()}
        ], lr=args.base_lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError()

    scheduler = CosineLRScheduler(
            optimizer,
            t_initial=args.max_epoch * n_iter_per_epoch,
            lr_min=args.base_lr * args.lr_factor,
            warmup_lr_init=args.base_lr * args.lr_factor,
            warmup_t=args.warmup_epochs * n_iter_per_epoch,
            cycle_limit=15,
            t_in_epochs=False,
    )
    return optimizer, scheduler


def resume_train(base_model, psnet_model, decoder,
                            dim_reducer1, dim_reducer2, optimizer, dim_reducer3,regressor_delta, segmenter,video_encoder, args):
    ckpt_path = os.path.join(args.experiment_path, 'best.pth')
    if not os.path.exists(ckpt_path):
        print('no checkpoint file from path %s...' % ckpt_path)
        return 0, 0, 0, 1000, 1000
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # parameter resume of base model
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    psnet_ckpt = {k.replace("module.", ""): v for k, v in state_dict['psnet_model'].items()}
    psnet_model.load_state_dict(psnet_ckpt)

    decoder_ckpt1 = {k.replace("module.", ""): v for k, v in state_dict['decoder1'].items()}
    decoder[0].load_state_dict(decoder_ckpt1)
    decoder_ckpt1 = {k.replace("module.", ""): v for k, v in state_dict['decoder2'].items()}
    decoder[1].load_state_dict(decoder_ckpt1)
    decoder_ckpt1 = {k.replace("module.", ""): v for k, v in state_dict['decoder3'].items()}
    decoder[2].load_state_dict(decoder_ckpt1)
    decoder_ckpt1 = {k.replace("module.", ""): v for k, v in state_dict['decoder4'].items()}
    decoder[3].load_state_dict(decoder_ckpt1)
    regressor_delta_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor_delta1'].items()}
    regressor_delta[0].load_state_dict(regressor_delta_ckpt)
    regressor_delta_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor_delta2'].items()}
    regressor_delta[1].load_state_dict(regressor_delta_ckpt)
    regressor_delta_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor_delta3'].items()}
    regressor_delta[2].load_state_dict(regressor_delta_ckpt)

    mask_encoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['segmenter'].items()}
    segmenter.load_state_dict(mask_encoder_ckpt)
    dim_reducer3_ckpt = {k.replace("module.", ""): v for k, v in state_dict['dim_reducer3'].items()}
    dim_reducer3.load_state_dict(dim_reducer3_ckpt)

    dim_reducer1_ckpt = {k.replace("module.", ""): v for k, v in state_dict['dim_reducer1'].items()}
    dim_reducer1.load_state_dict(dim_reducer1_ckpt)

    mask_encoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['video_encoder'].items()}
    video_encoder.load_state_dict(mask_encoder_ckpt)
    dim_reducer2_ckpt = {k.replace("module.", ""): v for k, v in state_dict['dim_reducer2'].items()}
    dim_reducer2.load_state_dict(dim_reducer2_ckpt)

    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

    # parameter
    start_epoch = state_dict['epoch'] + 1
    epoch_best_aqa = state_dict['epoch_best_aqa']
    rho_best = state_dict['rho_best']
    L2_min = state_dict['L2_min']
    RL2_min = state_dict['RL2_min']

    return start_epoch, epoch_best_aqa, rho_best, L2_min, RL2_min


def load_model(base_model, psnet_model, decoder,
                            dim_reducer1, dim_reducer2, dim_reducer3,regressor_delta, segmenter,video_encoder,args):
    ckpt_path = os.path.join(args.experiment_path, 'best.pth')
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path,map_location='cpu')

    # parameter resume of base model
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    psnet_ckpt = {k.replace("module.", ""): v for k, v in state_dict['psnet_model'].items()}
    psnet_model.load_state_dict(psnet_ckpt)

    decoder_ckpt1 = {k.replace("module.", ""): v for k, v in state_dict['decoder1'].items()}
    decoder[0].load_state_dict(decoder_ckpt1)
    decoder_ckpt1 = {k.replace("module.", ""): v for k, v in state_dict['decoder2'].items()}
    decoder[1].load_state_dict(decoder_ckpt1)
    decoder_ckpt1 = {k.replace("module.", ""): v for k, v in state_dict['decoder3'].items()}
    decoder[2].load_state_dict(decoder_ckpt1)
    decoder_ckpt1 = {k.replace("module.", ""): v for k, v in state_dict['decoder4'].items()}
    decoder[3].load_state_dict(decoder_ckpt1)
    regressor_delta_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor_delta1'].items()}
    regressor_delta[0].load_state_dict(regressor_delta_ckpt)
    regressor_delta_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor_delta2'].items()}
    regressor_delta[1].load_state_dict(regressor_delta_ckpt)
    regressor_delta_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor_delta3'].items()}
    regressor_delta[2].load_state_dict(regressor_delta_ckpt)

    mask_encoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['segmenter'].items()}
    segmenter.load_state_dict(mask_encoder_ckpt)
    dim_reducer3_ckpt = {k.replace("module.", ""): v for k, v in state_dict['dim_reducer3'].items()}
    dim_reducer3.load_state_dict(dim_reducer3_ckpt)

    dim_reducer1_ckpt = {k.replace("module.", ""): v for k, v in state_dict['dim_reducer1'].items()}
    dim_reducer1.load_state_dict(dim_reducer1_ckpt)

    dim_reducer2_ckpt = {k.replace("module.", ""): v for k, v in state_dict['dim_reducer2'].items()}
    dim_reducer2.load_state_dict(dim_reducer2_ckpt)
    mask_encoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['video_encoder'].items()}
    video_encoder.load_state_dict(mask_encoder_ckpt)


    epoch_best_aqa = state_dict['epoch_best_aqa']
    rho_best = state_dict['rho_best']
    L2_min = state_dict['L2_min']
    RL2_min = state_dict['RL2_min']
    print('ckpts @ %d epoch(rho = %.4f, L2 = %.4f , RL2 = %.4f)' % (epoch_best_aqa - 1, rho_best,  L2_min, RL2_min))
    return

class DualRandomHorizontalFlip:
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return [
                    img_.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img_ in img
                ], [ mask_.transpose(PIL.Image.FLIP_LEFT_RIGHT) for mask_ in mask]
            
        return img, mask

class DualCenterCrop:
    def __init__(self, size):
        self.trans = video_transforms.CenterCrop(size)

    def __call__(self, img, mask):
        cropped_img = self.trans(img)
        cropped_mask = self.trans(mask)

        return cropped_img, cropped_mask

class DualResize:
    def __init__(self, size):
        self.trans = video_transforms.Resize(size)

    def __call__(self, img, mask):
        return self.trans(img), self.trans(mask)


class DualRandomCrop:
    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, img, mask):
        im_w, im_h = img[0].size
        h, w = self.size 
        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)

        return  [img_.crop((x1, y1, x1+w, y1+h)) for img_ in img],\
              [mask_.crop((x1, y1, x1+w, y1+h)) for mask_ in mask]

class DualNormalize:
    def __init__(self, mean, std):
        self.trans =  video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.trans_mask = video_transforms.Normalize(mean=[0.5], std=[0.5])

    def __call__(self, img, mask):
        return self.trans(img), mask # self.trans_mask(mask)

class DualClipToTensor:
    def __init__(self):
        self.trans = volume_transforms.ClipToTensor()
        self.trans_mask = volume_transforms.ClipToTensor(channel_nb=1)
    def __call__(self, img, mask):
        return self.trans(img), self.trans_mask(mask)

class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask
