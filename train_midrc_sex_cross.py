import pickle
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import json
import numpy as np
import time
import copy
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import cv2
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from data_id_sex import get_id
from dataset import Dataset
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
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    return roc_auc_score(gt_np[:,0], pred_np[:,0])

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

                
    groupp = group0p+group1p 
    groupn = group0n+group1n
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
        A0a = cross_auc(torch.index_select(outputs_,0,torch.tensor(group0p)), torch.index_select(outputs_,0,torch.tensor(groupn)))
    except:
        A0a = 1
    try:
        A1a = cross_auc(torch.index_select(outputs_,0,torch.tensor(group1p)), torch.index_select(outputs_,0,torch.tensor(groupn)))
    except:
        A1a = 1
    try:
        Aa0 = cross_auc(torch.index_select(outputs_,0,torch.tensor(groupp)), torch.index_select(outputs_,0,torch.tensor(group0n)))
    except:
        Aa0 = 1
    try:
        Aa1 = cross_auc(torch.index_select(outputs_,0,torch.tensor(groupp)), torch.index_select(outputs_,0,torch.tensor(group1n)))
    except:
        Aa1 = 1
                

    group_num = [len(group0p),len(group0n),len(group1p),len(group1n)]
    
    return AUC, A00, A11, A0a, A1a, Aa0, Aa1, group_num




# def train_model(dataloaders,model, criterion, optimizer, scheduler, num_epochs=25):
def train_model(dataloaders,model, criterion, optimizer, num_epochs=25):
    since = time.time()
    fopen = open("/prj0129/mil4012/AREDS/accuracy_pfm_gender.txt", "w")
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
            groups = []
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            # Iterate over data.
            t = tqdm(enumerate(dataloaders[phase]),  desc='Loss: **** ', total=len(dataloaders[phase]), bar_format='{desc}{bar}{r_bar}')
            for batch_idx, (inputs, labels, group) in t:
                # if batch_idx == 0:
                #     continue
                # print(torch.isnan(inputs).sum())
                inputs = inputs.to(device)
                labels = labels.to(device)
                #print(inputs.shape, labels.shape)
                # print('the lables is',torch.unique(labels))
                if len(torch.unique(labels)) !=1 and len(np.unique(group)) != 1:
                    # print(len(torch.unique(labels)))
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)

                        gt = torch.cat((gt, labels), 0)
                        pred = torch.cat((pred, outputs.data), 0)
                        groups += group

                        # print('outputs shape',outputs.shape)
                        # print('labels shape', labels.shape)
                        # print('groups shape', group.shape)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    losses.update(loss.data.item(), inputs.size(0))
                    t.set_description('Loss: %.3f ' % (losses.avg))
            
            AUCs = compute_AUCs(gt, pred)
            AUROC_avg = AUCs
            AUC, A00, A11, A0a, A1a, Aa0, Aa1, group_num = group_auc(gt, pred, groups)
            
            if phase == "val":
                
                # scheduler.step(losses.avg)
                
                if best_AUROC_avg < AUROC_avg:
                    best_AUROC_avg = AUROC_avg
                    torch.save(model.state_dict(), "/prj0129/mil4012/MIDRC/weights/densenet121_sex_midrc_cross.pth")
                fopen.write('\nEpoch {} \t [{}] : \t {AUROC_avg:.3f}\n'.format(epoch, phase, AUROC_avg=AUROC_avg))
                fopen.write('{} \t {}\n'.format(CLASS_NAMES, AUCs))
                fopen.write('-' * 100)
                    
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, batch_idx + 1, len(dataloaders[phase]), loss=losses))
            print('{} : \t {AUROC_avg:.3f}'.format(phase, AUROC_avg=AUROC_avg))
            print('AUC',AUC)
            print('A00',A00)
            print('A11',A11)
            print('A0a',A0a)
            print('A1a',A1a)
            print('Aa0',Aa0)
            print('Aa1',Aa1)
            print('Group Num',group_num)
            
            fopen.flush()
    fopen.close()
    return model


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
    AUC, A00, A11, A0a, A1a, Aa0, Aa1, group_num = group_auc(gt, pred, groups)
    print('AUCs',AUCs)
    print('AUC',AUC)
    print('A00',A00)
    print('A11',A11)
    print('A0a',A0a)
    print('A1a',A1a)
    print('Aa0',Aa0)
    print('Aa1',Aa1)
    print('Group Num',group_num)
    pred1 = pred.cpu()
    pred2 = pred1.numpy()
    gt1 = gt.cpu()
    gt2 = gt1.numpy()
    np.savez('/prj0129/mil4012/MIDRC/Result/densenet121_race_midrc_cross_f5.npz', prediction=pred2, label=gt2, group=groups)
    np.savetxt('/prj0129/mil4012/MIDRC/Result/densenet121_sex_midrc_cross_f5.txt', pred2)
            
        

if __name__ == '__main__':
    
    fold = 5
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
    
    # #used imagenet
    # model_ft = models.densenet201(pretrained=True)
    # num_ftrs = model_ft.classifier.in_features
    # model_ft.classifier = nn.Sequential(
    #             nn.Linear(num_ftrs, N_CLASSES),
    #             nn.Sigmoid()
    #         )
    # # model_ft.classifier = nn.Linear(num_ftrs, N_CLASSES)
    # model_ft = model_ft.to(device)
    
    #used chexpert
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
    
    model_ft.classifier = nn.Sequential(
            nn.Linear(num_ftrs, N_CLASSES),
            nn.Sigmoid()
        )

    # model_ft.classifier = nn.Linear(num_ftrs, N_CLASSES)
    
    model_ft = model_ft.to(device)
    
    sc=torch.tensor([0.1,0.9]).to(device)
    # criterion = nn.CrossEntropyLoss(weight=sc)
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()
    
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)
    
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, 'min', patience=2, eps=1e-08, verbose=True)

    # model_ft = train_model(dataloaders, model_ft, criterion, optimizer_ft, exp_lr_scheduler,
    #                        num_epochs=20)
#     model_ft = train_model(dataloaders, model_ft, criterion, optimizer_ft,num_epochs=20)
    model_ft.load_state_dict(torch.load("/prj0129/mil4012/MIDRC/weights/densenet121_race_midrc_cross_f5.pth"))
    test_model(test_loader,model_ft)