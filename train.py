import os
import torch
import wandb
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch import nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from models.Densenet import densenet121
from Func import evaluate
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score,roc_auc_score
from config import train_root,val_root, batch_size,n_epoch,save_root,input_size,nw,lr,save_datainfo_root

opj = os.path.join

def main():
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(input_size),
                                     transforms.RandomHorizontalFlip(0.5),
                                     transforms.ColorJitter(brightness=0.2, contrast=0.2),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                          std=[0.229, 0.224, 0.225])]),

        "val": transforms.Compose([transforms.Resize(512),
                                   transforms.CenterCrop(input_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225])])}
    
    train_dataset = ImageFolder(root=train_root,
                                transform=data_transform["train"])
    
    val_dataset = ImageFolder(root=val_root,
                              transform=data_transform["val"])
    
    
    # 封装训练集
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=nw,
                              drop_last = False)

    # 封装验证集
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=nw,
                            drop_last = False)
    # # 各类别名称
    # class_names = train_dataset.classes
    # # 映射关系：类别 到 索引号
    # train_dataset.class_to_idx
    # 映射关系：索引号 到 类别
    idx_to_labels = {y:x for x,y in train_dataset.class_to_idx.items()}
    # 保存为本地的 npy 文件
    np.save(opj(save_datainfo_root,'idx_to_labels.npy'), idx_to_labels)
    np.save(opj(save_datainfo_root,'labels_to_idx.npy'), train_dataset.class_to_idx)

    print('映射关系:', train_dataset.class_to_idx) 
    print('训练集规格:', train_dataset[1][0].size())
    print('len of train_dataset: ',len(train_dataset))
    print('len of val_dataset: ',len(val_dataset))
    print('Len of train_loader: ',len(train_loader))
    print('Len of val_loader: ',len(val_loader))

    model = densenet121(drop_rate = 0.4)
    del model.classifier 
    block = nn.Sequential(nn.Linear(in_features=1024, out_features=512, bias=True),
                      nn.ReLU(inplace=True),nn.Dropout(0.5),
                      nn.Linear(in_features=512, out_features=128, bias=True),
                      nn.ReLU(inplace=True),
                      nn.Linear(in_features=128, out_features=2, bias=True),
                      )
    model.add_module('classifier',block)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), 'gpus')
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(),lr=lr,betas=(0.9, 0.9))
    # optimizer = optim.RAdam(model.parameters(),lr=lr,betas=(0.8, 0.9))
    # optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9)

    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss() 
    batch_n = len(train_loader)
    # #学习率降低策略
    # T_max是周期的一半
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,T_max = n_epoch*batch_n/7,eta_min=1e-6,verbose=True)
    
    #lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2,eta_min=1e-6)
    #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9,verbose=True)
    
    #lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,pct_start=0.3,total_steps = n_epoch*batch_n,div_factor=10,final_div_factor=1000,verbose=True)
    
    # lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=0.1,
    #                                           mode = 'exp_range', gamma = 0.9,
    #                                           step_size_up=200)
    

    epoch = 0
    best_roc_auc_score = 0.0
    # 训练日志-训练集
    df_train_log = pd.DataFrame()
    # 训练日志-测试集
    df_test_log = pd.DataFrame()
    # lr
    df_lr_log = pd.DataFrame()


    wandb.init(project='HPA', name=time.strftime('%m%d%H%M%S'))
    wandb.watch(model, log="gradients", log_freq=1000, log_graph=False)

    for epoch in range(1, n_epoch+1):
        print(f'Epoch {epoch}/{n_epoch}')
        ## 训练阶段
        model.train()
        loss_list = []
        labels_list = []
        preds_list = []
        batch_num = 0
        for images, labels in tqdm(train_loader): # 获得一个 batch 的数据和标注
            # 获得一个 batch 的数据和标注
            images = images.to(device)
            labels = labels.to(device)
            # 输入模型，执行前向预测
            outputs = model(images) 
            # 计算当前 batch 中，每个样本的平均交叉熵损失函数值
            loss = criterion(outputs, labels) 
            # 优化更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_num +=1
            # 获取当前 batch 的标签类别和预测类别
            _, preds = torch.max(outputs, 1) # 获得当前 batch 所有图像的预测类别
            preds = preds.cpu().numpy()
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)

            log_lr = {}
            log_lr['epoch'] = epoch
            log_lr['batch'] = batch_num
            log_lr['lr'] = optimizer.param_groups[0]['lr']
            df_lr_log = pd.concat([df_lr_log,pd.DataFrame([log_lr])],ignore_index=True)
            wandb.log(log_lr)
            #注意
            lr_scheduler.step()

        log_train = {}
        log_train['epoch'] = epoch
        log_train['train_loss'] = np.mean(loss_list)
        log_train['train_accuracy'] = accuracy_score(labels_list, preds_list)
        log_train['train_precision'] = precision_score(labels_list, preds_list, average='macro',zero_division=0)
        log_train['train_recall'] = recall_score(labels_list, preds_list, average='macro',zero_division=0)
        log_train['train_f1-score'] = f1_score(labels_list, preds_list, average='macro',zero_division=0)
        log_train['train_roc_auc_score'] = roc_auc_score(labels_list, preds_list, average='macro')
        df_train_log = pd.concat([df_train_log,pd.DataFrame([log_train])],ignore_index=True)
        wandb.log(log_train)
            
        ## 测试阶段
        model.eval()
        log_test = evaluate(model, device, criterion, val_loader,epoch)
        df_test_log = pd.concat([df_test_log,pd.DataFrame([log_test])],ignore_index=True)
        wandb.log(log_test)
          
        # 保存最新的最佳模型文件
        if log_test['test_roc_auc_score'] > best_roc_auc_score: 
            # 删除旧的最佳模型文件(如有)
            old_best_checkpoint_path = save_root+'/best-{:.3f}.pth'.format(best_roc_auc_score)
            if os.path.exists(old_best_checkpoint_path):
                os.remove(old_best_checkpoint_path)
            # 保存新的最佳模型文件
            best_roc_auc_score = log_test['test_roc_auc_score']
            new_best_checkpoint_path = save_root+'/best-{:.3f}.pth'.format(log_test['test_roc_auc_score'])

            if not os.path.exists(save_root):
                os.makedirs(save_root)

            torch.save(model.state_dict(), new_best_checkpoint_path)
            print('保存新的最佳模型', save_root+'/best-{:.3f}.pth'.format(best_roc_auc_score))

    save_root_dir = save_root+'/best-{:.3f}'.format(best_roc_auc_score)
    if not os.path.exists(save_root_dir):
        os.makedirs(save_root_dir)

    df_train_log.to_csv(save_root_dir+'/train_log.csv', index=False)
    df_test_log.to_csv(save_root_dir+'/val_log.csv', index=False)
    df_lr_log.to_csv(save_root_dir+'/lr_log.csv', index=False)



if __name__ == '__main__':

    main()