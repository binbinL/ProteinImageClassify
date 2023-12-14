import torch
import numpy as np
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score,roc_auc_score

def evaluate(model, device, criterion, test_loader,epoch):
    '''
    在整个测试集上评估，返回分类评估指标日志
    '''
    loss_list = []
    labels_list = []
    preds_list = []
    
    with torch.no_grad():
        for images, labels in test_loader: # 生成一个 batch 的数据和标注
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images) # 输入模型，执行前向预测

            # 获取整个测试集的标签类别和预测类别
            _, preds = torch.max(outputs, 1) # 获得当前 batch 所有图像的预测类别
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels) # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)
        
    log_test = {}
    log_test['epoch'] = epoch
    # 计算分类评估指标
    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro',zero_division=0)
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro',zero_division=0)
    log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro',zero_division=0)
    log_test['test_roc_auc_score'] = roc_auc_score(labels_list, preds_list, average='macro')

    return log_test
