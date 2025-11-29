+++
date = '2025-11-29T14:03:58+08:00'
draft = true
title = 'Change Detection'
author = 'BruceZhang'
+++

## 1. ChangeCLIP

论文：[ChangeCLIP: Remote sensing change detection with multimodal vision-language representation learning](https://www.sciencedirect.com/science/article/pii/S0924271624000042)

问题：传统深度学习方法更注重视觉表征学习，忽视了多模态数据的潜力。

解决：以 [CLIP](https://www.zhihu.com/tardis/zm/art/662365120?source_id=1003) 模型为基础，设计了遥感图像变化前后特征的具体提示。因此，这种方法能够构建一个富含多模先验的基础数据集，用于变化检测任务。

![ChangeCLIP](ChangCLIP.jpg "ChangeCLP")

![ChangeCLP_architecture](ChangeCLIP_arch.jpg "ChangeCLP_architecture")

代码：<https://github.com/dyzy41/ChangeCLIP>

## 2. ChangeMamba

论文：[ChangeMamba: Remote Sensing Change Detection with Spatio-Temporal State Space Model](https://arxiv.org/abs/2404.03425)

问题：传统 CNN 和 Transformer 用于 CD，但 CNN 受限于有限的感受野，无法捕捉像素间的长距离依赖性，使其在处理具有不同时空分辨率的复杂多时空场景时仍然难以做到；而 Transformers 计算量大，复杂度为$O(n^2)$，使其在大数据集上的训练和部署成本较高。

> CD tasks can be categorized into three types, namely binary CD (BCD), semantic CD (SCD), and building damage assessment (BDA)

解决：尝试将 [Mamba](https://www.ibm.com/cn-zh/think/topics/mamba-model) 架构应用于 CD 任务，并提出了若干时空状态空间模型（统称为 ChangeMamba）。ChangeMamba 基于最近提出的 [VMamba](https://blog.csdn.net/soaring_casia/article/details/136052041#:~:text=%E2%97%8FVMamba%EF%BC%8C%E4%B8%80%E7%A7%8D%E8%A7%86%E8%A7%89,CNN%E5%92%8CViT%E7%9A%84%E6%89%A9%E5%B1%95%E3%80%82&text=%E5%BC%95%E5%85%A5%E4%BA%86%E4%BA%A4%E5%8F%89%E6%89%AB%E6%8F%8F%E6%A8%A1%E5%9D%97%EF%BC%88CSM%EF%BC%89%EF%BC%8C%E8%A7%A3%E5%86%B3%E4%BA%861,%E5%85%A8%E5%B1%80%E6%84%9F%E5%8F%97%E9%87%8E%E7%9A%84%E7%89%B9%E6%80%A7%E3%80%82) 架构，该架构采用交叉扫描模块（CSM），能够向不同空间方向展开图像补丁，从而从图像中有效建模全局上下文信息。由于 CD 任务要求探测器从多时态图像中充分学习时空特征，我们设计了三种时空-时间关系建模机制，并自然地将它们与 Mamba 架构结合。

![ChangeMamba](ChangeMamba1.png "ChangeMamba")

代码：<https://github.com/ChenHongruixuan/MambaCD>