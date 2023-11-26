from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import copy
from util import AverageMeter
from util import set_optimizer, save_model
from data_id_sex import get_id
from dataset import Dataset
import types
path = '/prj0129/mil4012/MIDRC' 
path_r = '/prj0129/mil4012/MIDRC/result_p/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_train_test_p_id(patient_list, fold, total_num_fold):
    num = len(patient_list)
    test_num = num // total_num_fold
    
    if fold == total_num_fold:
        test_name = patient_list[((fold-1) * test_num):]
        train_name = patient_list[0:((fold-1) * test_num)]
    else:
        test_name = patient_list[((fold-1) * test_num):fold * test_num]
        train_name = np.concatenate((patient_list[0:((fold-1) * test_num)], patient_list[(fold * test_num):]), axis=0)
    

    validation_name = train_name[int(0.8*len(train_name)):]


    train_name = train_name[0:(len(train_name)-len(validation_name))]

    return train_name, validation_name, test_name



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.1, base_temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels, groups, device):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        features = features.view(features.shape[0], -1)

        batch_size = features.shape[0]

        labels = labels.view(1, -1)[0]
        if len(labels) != batch_size or len(groups) != batch_size:
            raise ValueError('Num of labels or groups does not match num of features')
            
        pos_mask = torch.zeros(batch_size, batch_size)
        for i in range(batch_size):
            for j in range(batch_size):
                if labels[i] == labels[j]:
                    pos_mask[i][j] = 1
        
        neg_mask = torch.zeros(batch_size, batch_size)
        for i in range(batch_size):
            for j in range(batch_size):
                if (labels[i] != labels[j]) or (labels[i] == labels[j]):
                    neg_mask[i][j] = 1

        # features = features[pos_mask.sum(1)>0]
        # neg_mask = neg_mask[pos_mask.sum(1)>0].float().to(device)
        # pos_mask = pos_mask[pos_mask.sum(1)>0].float().to(device)

        contrast_feature = features
        anchor_feature = contrast_feature

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        anchor_dot_contrast = anchor_dot_contrast[pos_mask.sum(1)>0]
        neg_mask = neg_mask[pos_mask.sum(1)>0].float().to(device)
        pos_mask = pos_mask[pos_mask.sum(1)>0].float().to(device)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast*neg_mask, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases

        # compute log_prob
        exp_logits = torch.exp(logits * neg_mask) * neg_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        


        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, -1).mean()

        return loss

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs')
    parser.add_argument('--cuda', type=str, default='0',
                        help='cuda')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--trial', type=str, default='1',
                        help='id for recording multiple runs')

    opt = parser.parse_args()
    
    opt.device = torch.device('cuda:'+ opt.cuda)


    # set the path according to the environment
    opt.model_path = 'save'

#     opt.model_name = 'sex_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
#         format(opt.learning_rate, opt.weight_decay, opt.batch_size, opt.temp, opt.trial)
    
    opt.model_name = 'sex_original_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.learning_rate, opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    return opt


def set_train_loader(opt,train_path,train_labels,train_groups):
        
    data_transforms = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.Resize(256),
        transforms.CenterCrop(224),
#         transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = Dataset(train_path,train_labels,groups=train_groups,transform = data_transforms)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    

    return train_loader

def set_val_loader(opt,val_path,val_labels,val_groups):
    
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = Dataset(val_path,val_labels,groups=val_groups,transform = data_transforms)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True)

    return val_loader


def set_model(opt):
    model_ft = models.densenet121(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Sequential(
                nn.Linear(num_ftrs, 14),
                nn.Sigmoid()
            )
    checkpoint = torch.load("/prj0129/mil4012/glaucoma/NIH-chest-x-ray/CXR8/SupCon/m-30012020-104001.pth.tar", map_location=torch.device('cpu'))
    for i in list(checkpoint['state_dict'].keys()):
        checkpoint['state_dict'][i[12:]] = checkpoint['state_dict'].pop(i)
    model_ft.load_state_dict(checkpoint['state_dict'])

    model_ft.classifier = nn.Linear(num_ftrs, 128)

    model_ft = model_ft.to(opt.device)
    
    criterion = SupConLoss(temperature=opt.temp)

    return model_ft, criterion

def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels, groups) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.to(opt.device)
        labels = labels.to(opt.device)
        
        bsz = labels.shape[0]

        # compute loss
        features = model(images)
        
        loss = criterion(features, labels, groups, opt.device)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()

        loss.backward()
        
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def val(val_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    
    with torch.no_grad():
        for idx, (images, labels, groups) in enumerate(val_loader):
            data_time.update(time.time() - end)

            images = images.to(opt.device)
            labels = labels.to(opt.device)

            bsz = labels.shape[0]

            # compute loss
            features = model(images)

            loss = criterion(features, labels, groups, opt.device)

            # update metric
            losses.update(loss.item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx + 1) % opt.print_freq == 0:
                print('Val: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                       epoch, idx + 1, len(val_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses))
                sys.stdout.flush()

    return losses.avg

opt = parse_option()


fold = 1
total_num_fold = 5
    
train_sampler = None
batch_size = 128
workers = 4
N_CLASSES = 1
CLASS_NAMES = 'MIDRC'
    
data_path=os.path.join(path,'data_resize/')
label_path=os.path.join(path,'filtered_final1.csv')
    
label_path1 = os.path.join(path,'Patient_list.csv')
tmp = np.loadtxt(label_path1, dtype=np.str_, delimiter=",")
    
tmp = tmp[1:] 
train_name, validation_name, test_name = get_train_test_p_id(tmp, fold, total_num_fold)
    
    
train_path, train_labels, train_groups = get_id(data_path,label_path,data_id = train_name)
val_path, val_labels, val_groups = get_id(data_path,label_path,data_id = validation_name)
# test_path, test_labels, test_groups = get_id(data_path,label_path,data_id = test_name)
    


# build data loader
train_loader = set_train_loader(opt,train_path,train_labels,train_groups)
val_loader = set_val_loader(opt,val_path,val_labels,val_groups)


# build model and criterion
model, criterion = set_model(opt)

# build optimizer
optimizer = set_optimizer(opt, model)


# training routine
for epoch in range(1, opt.epochs + 1):

    # train for one epoch
    time1 = time.time()
    train_loss = train(train_loader, model, criterion, optimizer, epoch, opt)
    val_loss = val(val_loader, model, criterion, optimizer, epoch, opt)
    time2 = time.time()
    print('Epoch {}, Total time {:.2f}'.format(epoch, time2 - time1))

    print('Train loss', train_loss)
    print('Val loss', val_loss)

    if epoch % opt.save_freq == 0:
        save_file = os.path.join(
            opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        save_model(model, optimizer, opt, epoch, save_file)

# save the last model
save_file = os.path.join(
    opt.save_folder, 'last.pth')
save_model(model, optimizer, opt, opt.epochs, save_file)