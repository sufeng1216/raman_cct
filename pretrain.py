import pandas as pd
from time import time
import numpy as np
import os
import torch
from vit_pytorch.cct import CCT
from datasets import spectral_dataloader
from training import run_epoch
from torch import optim
from timm.scheduler import CosineLRScheduler
from datetime import datetime

t00 = time()
X_fn = './data/X_reference.npy'
y_fn = './data/y_reference.npy'
X = np.load(X_fn)
y = np.load(y_fn)
print(X.shape, y.shape)

os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(0)
cuda = torch.cuda.is_available()

net = CCT(
    img_size=(25, 40),
    embedding_dim=128,
    n_input_channels=1,
    n_conv_layers=1,
    kernel_size=3,
    stride=2,
    padding=3,
    pooling_kernel_size=3,
    pooling_stride=2,
    pooling_padding=1,
    num_layers=7,
    num_heads=4,
    mlp_ratio=2.,
    num_classes=30,
    positional_embedding='learnable',  # ['sine', 'learnable', 'none']
)

# net = torch.load('SOTA/83.6/model_vit_finetuned_83.6.pkl')
if cuda: net.cuda()
# Fine-tuning


# Train/val split
p_val = 0.1
n_val = int(60000 * p_val)
idx_tr = list(range(60000))
np.random.shuffle(idx_tr)
idx_val = idx_tr[:n_val]
idx_tr = idx_tr[n_val:]


now = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
epochs = 300  # Change this number to ~30 for full training
batch_size = 64

# Set up Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=55e-5, betas=(0.5, 0.999))
scheduler = CosineLRScheduler(optimizer=optimizer,
                              t_initial=200,
                              lr_min=1e-5,
                              warmup_t=10,
                              warmup_lr_init=0.00001
                              )

# Set up dataloaders
dl_tr = spectral_dataloader(X, y, idxs=idx_tr,
                            batch_size=batch_size, shuffle=True)
dl_val = spectral_dataloader(X, y, idxs=idx_val,
                             batch_size=batch_size, shuffle=False)

# Fine-tune for first fold
best_val = 0
df = pd.DataFrame(columns=['epoch', 'train_Loss', 'train_acc', 'val_Loss', 'val_acc'])  # 列名
df.to_csv("./results/pr_" + now + ".csv", index=False)  # 路径可以根据需要更改
no_improvement = 0
max_no_improvement = 15
# print('Starting fine-tuning!')
for epoch in range(epochs):
    print(' Epoch {}: '.format(epoch))
    scheduler.step(epoch)
    # Train
    acc_tr, loss_tr = run_epoch(epoch, net, dl_tr, cuda,
                                training=True, optimizer=optimizer)
    print('  Train acc: {:0.2f}'.format(acc_tr))
    # Val
    acc_val, loss_val = run_epoch(epoch, net, dl_val, cuda,
                                  training=False, optimizer=optimizer)
    test_list = [epoch, loss_tr, acc_tr, loss_val, acc_val]
    # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
    data = pd.DataFrame([test_list])
    # 3）将数据写入csv文件
    data.to_csv("./results/pr_" + now + ".csv", mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
    print('  Val acc  : {:0.2f}'.format(acc_val))
    # Check performance for early stopping
    if acc_val > best_val or epoch == 0:
        best_val = acc_val
    #     no_improvement = 0
    # else:
    #     no_improvement += 1
    # if no_improvement >= max_no_improvement:
    #     print('Finished after {} epochs!'.format(epoch))
    #     break
    torch.save(net.state_dict(), "./weights/model-{}.pth".format(epoch))
print('best val: {:0.2f}'.format(best_val))
# torch.save(net,'model.pkl')
# torch.save(net.state_dict(),'model_state_dict.pkl')
