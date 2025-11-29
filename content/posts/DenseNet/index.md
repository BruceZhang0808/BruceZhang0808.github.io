---
title: "DenseNet"
subtitle: "Learning Hugo + LoveIt"
date: 2025-07-31T10:00:00+02:00
draft: false
tags: ["hugo", "loveit", "learning"]
categories: ["blog"]
author: "Your Name"

summary: "This is my first post on my personal site using Hugo LoveIt theme. Exploring markdown, code blocks, images, and shortcodes."
---

# DenseNet网络学习笔记

1. 组成
   
    <figure>
    <img src="denseBlock.jpg" alt="Dense block" style="width:100%; height:auto; display:block;">
    <figcaption style="text-align:center;">
        A 5-layer dense block with a growth rate of k = 4. Each layer takes all preceding feature-maps as input.
    </figcaption>
    </figure>

2. Dense Layer
   
    一个 Dense Layer 内部结构：BN-ReLU-Conv1x1 -> BN-ReLU-Conv3x3，输出通道数为 $k$ 的特征图（其中 $k$ 为增长率 growth_rate，是DenseNet里面重要的超参数，一会介绍）。

3. Dense Block
   
    一个 Dense Block 是 $n$ 个 Dense Layer 堆叠而成的，第 $i$ 个 Dense Layer的输入特征图是原始输入数据+所有前 $i-1$ 层输出的特征图拼接后的整体，因此输入的通道数为 $c_0 + (i-1)k$，其中 $c_0$ 为原始数据通道数。最终 Dense Block 的输出，也是将当前所有层的的输出特征图拼接而成的整体，因此通道数为 $c_0 + nk$ 

4. Transition

    相邻两个 Dense Block 之间通过 Transition 过渡层相连，结构为 BN-ReLU-Conv1x1-AvgPool2x2，用于压缩数据大小

5. 最终组合

    <figure>
    <img src="./denseNet.jpg" alt="DenseNet" style="width:100%; height:auto; display:block;">
    <figcaption style="text-align:center;">
        A deep DenseNet with three dense blocks. 
    </figcaption>
    </figure>

文件`densnet.py`实现
   
   ```python
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from collections import OrderedDict
   
   class DenseLayer(nn.Module):
      def __init__(self, in_channels, growth_rate, bn_size=4):
          super().__init__()
          inter_channels = bn_size * growth_rate  # typical bottleneck factor
          self.bn1 = nn.BatchNorm2d(in_channels)
          self.relu1 = nn.ReLU(inplace=True)
          self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=False)
   
          self.bn2 = nn.BatchNorm2d(inter_channels)
          self.relu2 = nn.ReLU(inplace=True)
          self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
   
      def forward(self, x):
          out = self.conv1(self.relu1(self.bn1(x)))
          out = self.conv2(self.relu2(self.bn2(out)))
          return out  # shape: (batch, growth_rate, h, w)


   class DenseBlock(nn.Module):
       """Stack of dense layers; each layer sees concatenation of previous features"""
       def __init__(self, num_layers, in_channels, growth_rate, bn_size=4):
           super().__init__()
           self.layers = nn.ModuleList()
           for i in range(num_layers):
               layer = DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size)
               self.layers.append(layer)

       def forward(self, x):
           features = [x]
           for layer in self.layers:
               new_feat = layer(torch.cat(features, dim=1))
               features.append(new_feat)
           return torch.cat(features, dim=1)  # returns concatenated features

   class Transition(nn.Module):
       """Transition layer reduces channels and downsamples by 2 (avg pool)"""
       def __init__(self, in_channels, out_channels):
           super().__init__()
           self.bn = nn.BatchNorm2d(in_channels)
           self.relu = nn.ReLU(inplace=True)
           self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
           self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

       def forward(self, x):
           out = self.conv(self.relu(self.bn(x)))
           out = self.pool(out)
           return out

   class DenseNet(nn.Module):
       def __init__(self, growth_rate=12, block_config=(6,6,6), num_init_features=24, bn_size=4, num_classes=10):
           """
           block_config: tuple of number of layers in each dense block, e.g. (6,6,6) for CIFAR
           num_init_features: initial conv out channels; for CIFAR we often use smaller number than ImageNet
           """
           super().__init__()
           # initial conv (CIFAR size images 32x32)
           self.features = nn.Sequential(OrderedDict([
               ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
               ('bn0', nn.BatchNorm2d(num_init_features)),
               ('relu0', nn.ReLU(inplace=True)),
           ]))

           num_features = num_init_features
           # Dense blocks + transitions
           for i, num_layers in enumerate(block_config):
               block = DenseBlock(num_layers=num_layers, in_channels=num_features, growth_rate=growth_rate, bn_size=bn_size)
               self.features.add_module(f'denseblock{i+1}', block)
               num_features = num_features + num_layers * growth_rate
    
               if i != len(block_config) - 1:
                   # reduce channels (compression) typically by 0.5
                   out_features = num_features // 2
                   trans = Transition(num_features, out_features)
                   self.features.add_module(f'transition{i+1}', trans)
                   num_features = out_features
    
           # final batchnorm
           self.features.add_module('bn_final', nn.BatchNorm2d(num_features))
    
           # classifier
           self.classifier = nn.Linear(num_features, num_classes)
    
           # weight initialization (recommended)
           for m in self.modules():
               if isinstance(m, nn.Conv2d):
                   nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
               elif isinstance(m, nn.BatchNorm2d):
                   nn.init.constant_(m.weight, 1)
                   nn.init.constant_(m.bias, 0)
               elif isinstance(m, nn.Linear):
                   nn.init.constant_(m.bias, 0)
    
       def forward(self, x):
           features = self.features(x)
           out = F.relu(features, inplace=True)
           # global pooling to 1x1 then flatten
           out = F.adaptive_avg_pool2d(out, (1,1))
           out = torch.flatten(out, 1)
           out = self.classifier(out)
           return out

```

文件`main.py`实现

```python
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from densenet import DenseNet     

# -------------------------
# 超参数 / 配置
# -------------------------
BATCH_SIZE = 128
LR = 0.1
EPOCHS = 100
WEIGHT_DECAY = 5e-4
MILESTONES = [60, 80]   

SAVE_DIR = "./models"
SAVE_PATH = os.path.join(SAVE_DIR, "densenet.pth")
PRINT_EVERY_BATCHES = 40  

os.makedirs(SAVE_DIR, exist_ok=True)

label_names = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')

# -------------------------
# 设备与随机种子
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
import numpy as np
np.random.seed(seed)

# 可选：CUDNN 设置（提高速度但可能影响严格可复现）
torch.backends.cudnn.benchmark = True

# -------------------------
# DataLoader（注意 test 不要 drop_last）
# -------------------------
training_data = datasets.CIFAR10(
    root="../dataset",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = datasets.CIFAR10(
    root="../dataset",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=torch.cuda.is_available(), drop_last=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=torch.cuda.is_available(), drop_last=False)

# -------------------------
# 模型 / 损失 / 优化器 / scheduler
# -------------------------
model = DenseNet().to(device)

criterion = nn.CrossEntropyLoss()   # 无需 .to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
# 推荐的学习率调度器（任选其一）
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.1)
# 或者： scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# -------------------------
# 记录（用于画损失曲线）
# -------------------------
train_losses = []
train_accs = []
best_val_acc = 0.0

# -------------------------
# 训练函数
# -------------------------
def train():
    global best_val_acc
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        # tqdm库用于绘制进度显示，美化终端输出
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {epoch}/{EPOCHS}")
        batch_count = 0
        for i, (inputs, labels) in pbar:
            batch_count += 1
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


            # stats
            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_samples += labels.size(0)

            # update progress bar
            if i % PRINT_EVERY_BATCHES == 0 or i == len(train_loader):
                avg_loss = running_loss / batch_count   # 平均 per-batch loss since epoch start or since reset
                acc = running_correct / running_samples
                pbar.set_postfix({'avg_loss': f"{avg_loss:.4f}", 'acc': f"{acc:.4f}", 'lr': optimizer.param_groups[0]['lr']})

        # 每 epoch 结束调整 lr（如果使用 scheduler）
        scheduler.step()

        # 记录 epoch 指标
        epoch_loss = running_loss / batch_count
        epoch_acc = running_correct / running_samples
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # 验证/测试当前模型并保存最优
        val_acc = validate()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_acc': val_acc,
            }, SAVE_PATH)
            print(f"Saved new best model (val_acc={val_acc:.4f}) to {SAVE_PATH}")

# -------------------------
# 验证函数（返回 accuracy）
# -------------------------
def validate():
    model.eval()
    correct = 0
    total = 0
    incorrect_examples = []  # 我们只在内存允许下保存少量示例（CPU 端）
    max_saved = 100  # 最多保存 100 个错误样例，避免内存暴涨

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(inputs)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 保存错误样例（移动到 CPU 并 detach，以免占 GPU 显存）
            if len(incorrect_examples) < max_saved:
                mis_mask = preds != labels
                if mis_mask.any():
                    idxs = mis_mask.nonzero(as_tuple=True)[0]
                    for idx in idxs:
                        if len(incorrect_examples) >= max_saved:
                            break
                        img = inputs[idx].detach().cpu()  # TODO: 反归一化 -> 转 PIL / numpy 更方便可视化
                        pred_label = label_names[preds[idx].item()]
                        true_label = label_names[labels[idx].item()]
                        incorrect_examples.append((img, pred_label, true_label))

    acc = correct / total if total > 0 else 0.0
    print(f"Validation accuracy: {acc:.4f} ({correct}/{total})")
    # 如果你想在外面使用 incorrect_examples，可以返回它： return acc, incorrect_examples
    return acc

# -------------------------
# 主逻辑
# -------------------------
if __name__ == "__main__":
    start = time.time()
    try:
        train()
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "densenet_interrupt.pth"))
    finally:
        elapsed = time.time() - start
        print(f"Total time: {elapsed/60:.2f} minutes")
        # 最后在测试集上评估一次（并打印错误样本）
        final_acc = validate()
        print(f"Final validation accuracy: {final_acc:.4f}")
```
