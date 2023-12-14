"""config"""
train_root = '/data/lwb/data/HPA/train/mix'
val_root = '/data/lwb/data/HPA/validation/mix'
save_root = '/data/lwb/WorkSpace/LLPS/HPA/ImageClassify/checkpoint'
save_datainfo_root = '/data/lwb/WorkSpace/LLPS/HPA/ImageClassify/utils'
input_size = 384    # 裁剪图片大小
batch_size = 32    # 一次训练所选取的样本数
lr = 1e-2             # 学习率
n_epoch = 120          # 训练次数
nw = 4
