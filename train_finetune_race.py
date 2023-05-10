from __future__ import print_function, division
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import sys
import copy
from data_id_race import get_id
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


fold = 1
total_num_fold = 5
    
train_sampler = None
batch_size = 256
workers = 4
N_CLASSES = 1
CLASS_NAMES = ['MIDRC']
    
data_path=os.path.join(path,'data_resize/')
label_path=os.path.join(path,'filtered_final1.csv')
    
label_path1 = os.path.join(path,'Patient_list.csv')
tmp = np.loadtxt(label_path1, dtype=np.str_, delimiter=",")
    
tmp = tmp[1:] 
train_name, validation_name, test_name = get_train_test_p_id(tmp, fold, total_num_fold)
    
    
train_path, train_labels, train_groups = get_id(data_path,label_path,data_id = train_name)
val_path, val_labels, val_groups = get_id(data_path,label_path,data_id = validation_name)
test_path, test_labels, test_groups = get_id(data_path,label_path,data_id = test_name)



data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(10),
        # transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
    
train_dataset = Dataset(train_path,train_labels,groups=train_groups,transform = data_transforms["train"])
val_dataset = Dataset(val_path,val_labels,groups=val_groups,transform = data_transforms["val"])
test_dataset = Dataset(test_path,test_labels,groups=test_groups,transform = data_transforms["val"])
   
    
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), 
                          num_workers=workers, pin_memory=True, sampler=train_sampler)
        
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
#                                            num_workers=workers, pin_memory=True, sampler=train_sampler)
    
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=(train_sampler is None), 
                        num_workers=workers, pin_memory=True, sampler=train_sampler)
    
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=workers, pin_memory=True, sampler=train_sampler)


dataloaders = {"train": train_loader, "val": val_loader}



#image_name, image, classes = next(iter(train_loader))

class Counter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_AUCs(gt, pred):
    N_CLASSES = 1
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

def cross_auc(R_a_0, R_b_1): 
    scores = np.array(list(R_a_0.cpu().numpy()) + list(R_b_1.cpu().numpy()))
    y_true = np.zeros(len(R_a_0)+len(R_b_1))
    y_true[0:len(R_a_0)] = 1 # Pr[ LEFT > RIGHT]; Y = 1 is the left (A0)
    return roc_auc_score(y_true, scores)

def group_auc(labels, outputs, groups):
    
    group0p = []
    group0n = []
    group1p = []
    group1n = []
    group2p = []
    group2n = []

    for i in range(len(labels)):
        if groups[i] == 0:
            if labels[i][0] == 1:
                group0p.append(i)
            if labels[i][0] == 0:
                group0n.append(i)
        if groups[i] == 1:
            if labels[i][0] == 1:
                group1p.append(i)
            if labels[i][0] == 0:
                group1n.append(i)
        if groups[i] == 2:
            if labels[i][0] == 1:
                group2p.append(i)
            if labels[i][0] == 0:
                group2n.append(i)

                
    groupp = group0p+group1p+group2p 
    groupn = group0n+group1n+group2n
    outputs_ = outputs.clone().detach().cpu()
    
    try:
        AUC = cross_auc(torch.index_select(outputs_,0,torch.tensor(groupp)), torch.index_select(outputs_,0,torch.tensor(groupn)))
    except:
        AUC = 1
    try:
        A00 = cross_auc(torch.index_select(outputs_,0,torch.tensor(group0p)), torch.index_select(outputs_,0,torch.tensor(group0n)))
    except:
        A00 = 1
    try:
        A11 = cross_auc(torch.index_select(outputs_,0,torch.tensor(group1p)), torch.index_select(outputs_,0,torch.tensor(group1n)))
    except:
        A11 = 1
    try:
        A22 = cross_auc(torch.index_select(outputs_,0,torch.tensor(group2p)), torch.index_select(outputs_,0,torch.tensor(group2n)))
    except:
        A22 = 1
    try:
        A0a = cross_auc(torch.index_select(outputs_,0,torch.tensor(group0p)), torch.index_select(outputs_,0,torch.tensor(groupn)))
    except:
        A0a = 1
    try:
        A1a = cross_auc(torch.index_select(outputs_,0,torch.tensor(group1p)), torch.index_select(outputs_,0,torch.tensor(groupn)))
    except:
        A1a = 1
    try:
        A2a = cross_auc(torch.index_select(outputs_,0,torch.tensor(group2p)), torch.index_select(outputs_,0,torch.tensor(groupn)))
    except:
        A2a = 1
    try:
        Aa0 = cross_auc(torch.index_select(outputs_,0,torch.tensor(groupp)), torch.index_select(outputs_,0,torch.tensor(group0n)))
    except:
        Aa0 = 1
    try:
        Aa1 = cross_auc(torch.index_select(outputs_,0,torch.tensor(groupp)), torch.index_select(outputs_,0,torch.tensor(group1n)))
    except:
        Aa1 = 1
    try:
        Aa2 = cross_auc(torch.index_select(outputs_,0,torch.tensor(groupp)), torch.index_select(outputs_,0,torch.tensor(group2n)))
    except:
        Aa2 = 1
                

    group_num = [len(group0p),len(group0n),len(group1p),len(group1n),len(group2p),len(group2n)]
    
    return AUC, A00, A11, A22, A0a, A1a, A2a, Aa0, Aa1, Aa2, group_num



def test_model(test_loader,model):
    model.eval()
    gt = torch.FloatTensor().to(device)
    pred = torch.FloatTensor().to(device)
    groups = []
    with torch.no_grad():
        for batch_idx, (inputs, labels, group) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            gt = torch.cat((gt, labels), 0)
            pred = torch.cat((pred, outputs.data), 0)
            groups += group
    AUCs = compute_AUCs(gt, pred)
    AUC, A00, A11, A22, A0a, A1a, A2a, Aa0, Aa1, Aa2, group_num = group_auc(gt, pred, groups)
    
    print('AUCs',AUCs)
    print('AUC',AUC)
    print('A00',A00)
    print('A11',A11)
    print('A22',A22)
    print('A0a',A0a)
    print('A1a',A1a)
    print('A2a',A2a)
    print('Aa0',Aa0)
    print('Aa1',Aa1)
    print('Aa2',Aa2)
    print('Group Num',group_num)
    pred1 = pred.cpu()
    pred2 = pred1.numpy()

    np.savetxt('/prj0129/mil4012/MIDRC/Result/race_finetune_epoch_0_0.03new1.txt', pred2)

fopen = open("/prj0129/mil4012/AREDS/finetune_age_001_1.txt", "w")

model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 128)

checkpoint = torch.load("/prj0129/mil4012/MIDRC/save/race_lr_0.0001_decay_0.0001_bsz_256_temp_0.03_trial_1/last.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])

model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, N_CLASSES),
            nn.Sigmoid()
        )

model = model.to(device)
# print(model)

criterion = nn.BCELoss()

# Observe that all parameters are being optimized
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0, eps=1e-08, verbose=True)

num_epochs=1

since = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_AUROC_avg = 0.0
losses = Counter()

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 100)



    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        gt = torch.FloatTensor().to(device)
        pred = torch.FloatTensor().to(device)
        losses.reset()

        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        # Iterate over data.
        t = tqdm(enumerate(dataloaders[phase]),  desc='Loss: **** ', total=len(dataloaders[phase]), bar_format='{desc}{bar}{r_bar}')
        
        for batch_idx, (inputs, labels, group) in t:
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(inputs.shape, labels.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)

                gt = torch.cat((gt, labels), 0)
                pred = torch.cat((pred, outputs.data), 0)

                #print(outputs.shape)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            losses.update(loss.data.item(), inputs.size(0))
            t.set_description('Loss: %.3f ' % (losses.avg))

        AUCs = compute_AUCs(gt, pred)
        AUROC_avg = np.array(AUCs).mean()


        if phase == "val":
            torch.save(model.state_dict(), "/prj0129/mil4012/MIDRC/save/race_lr_0.0001_decay_0.0001_bsz_256_temp_0.03_trial_1/finetune_epoch_"+str(epoch)+"new1.pth")
            fopen.write('\nEpoch {} \t [{}] : \t {AUROC_avg:.3f}\n'.format(epoch, phase, AUROC_avg=AUROC_avg))
            for i in range(N_CLASSES):
                fopen.write('{} \t {}\n'.format(CLASS_NAMES[i], AUCs[i]))
            fopen.write('-' * 100)

        print('Epoch: [{0}][{1}/{2}]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
               epoch, batch_idx + 1, len(dataloaders[phase]), loss=losses))
        print('{} : \t {AUROC_avg:.3f}'.format(phase, AUROC_avg=AUROC_avg))

        #if phase == "val":
        #    scheduler.step(losses.avg)

        fopen.flush()
fopen.close()

model.load_state_dict(torch.load("/prj0129/mil4012/MIDRC/save/race_lr_0.0001_decay_0.0001_bsz_256_temp_0.03_trial_1/finetune_epoch_0new1.pth"))
test_model(test_loader,model)