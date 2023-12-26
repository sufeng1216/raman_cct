from torch.autograd import Variable
from torch import nn
import torch
import numpy as np


def run_epoch(epoch, model, dataloader, cuda, training=False, optimizer=None):
    if training:
        model.train()
    else:
        model.eval()  # 评估模式：batchNorm层和dropout层有所不同
    # In this mode, dropout layers are deactivated (i.e., they don't randomly drop out neurons),
    # and batch normalization layers use the precomputed running statistics
    # instead of computing statistics from the current mini-batch.
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if cuda: inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets.long())
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        if training:
            optimizer.zero_grad()  # 梯度会累积，每次epoch梯度清零
            loss.backward()  # 计算loss梯度，反向传播得到可调整的参数使loss最小化，The gradients are stored in the parameter objects.
            optimizer.step()  # 用上一步计算好的梯度更新模型参数

        total_loss += loss.item()  # 交叉熵损失默认reduction是mean，return 标量, 把batch里的loss相加
        _, predicted = torch.max(outputs.data, 1)  # 返回output里dim=1?的最大值和索引，当括号里只有一个参数input时返回最大值
        total += targets.size(0)  # 累计批样本总数
        if cuda:
            correct += predicted.eq(targets.data).cpu().sum().item()  # 累计正确的预测数量，为什么当用了cuda时要在cpu上计算这段？
        else:
            correct += predicted.eq(targets.data).sum().item()
    acc = 100 * correct / total
    avg_loss = total_loss / total  # calculates the average loss per sample
    return acc, avg_loss


def get_predictions(model, dataloader, cuda, get_probs=False):
    preds = []
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if cuda: inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets.long())
        outputs = model(inputs)
        if get_probs:
            probs = torch.nn.functional.softmax(outputs, dim=1)  # 因为是batch, 在dim=1上对数据进行softmax操作，得到概率
            if cuda:
                probs = probs.data.cpu().numpy()  # 转到cpu上计算
            else:
                probs = probs.data.numpy()
            preds.append(probs)
        else:
            _, predicted = torch.max(outputs.data, 1)
            if cuda: predicted = predicted.cpu()
            preds += list(predicted.numpy().ravel())
    if get_probs:
        return np.vstack(preds)  # 垂直堆叠，Stack the probability arrays vertically to obtain a 2D array.
    else:
        return np.array(preds)
