---
title: "Python for data science"
date: '2025-12-14'
draft: false
tags: ["learning", "Python"]
categories: ["notes"]
author: "Bruce Zhang"
summary: "用一些例子学习 Python 的数据科学库"
---

## 马氏距离

在学习马氏距离后，照着下面这个文章 [通俗易懂：马氏距离（举例）](https://zhuanlan.zhihu.com/p/674071134) 自己动手写了一下，体会到了协方差矩阵的特征分解是如何将数据的不同尺度统一起来的，从而转换为同一尺度下的欧式距离。

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'Noto Sans SC'
plt.rcParams['axes.unicode_minus'] = False

# 最后一个是离群值
heights = [175, 180, 168, 178, 182, 185, 177, 163, 190, 173, 182, 191, 167, 165, 171, 172, 175]
weights = [72, 78, 62, 70, 77, 88, 75, 46, 88, 71, 81, 90, 52, 50, 60, 65, 160]

# heights = [175, 180, 168, 178, 182, 185]
# weights = [72, 78, 62, 70, 77, 88]

x1 = np.array(heights, dtype=np.float64)
x2 = np.array(weights, dtype=np.float64)
X = np.stack((x1, x2))


def plot(X: np.ndarray):
    """绘制散点图
    
    :param X: shape=(2, n)
    :type X: np.ndarray
    """
    fig, ax = plt.subplots()
    x1, x2 = X[0,:], X[1,:]
    ax.scatter(x1, x2)
    ax.grid(True, alpha=0.5, ls='--')
    ax.set_xlabel(r'$x_1$', loc='right')
    ax.set_ylabel(r'$x_2$', rotation=0, loc='top', labelpad=-10)
    ax.set_aspect('equal', adjustable='datalim')
    if X.mean() < 1e-9:
        ax.spines["left"].set_position(("data", 0))
        ax.spines["bottom"].set_position(("data", 0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.show()

mu = X.mean(axis=1, keepdims=True)
X_regu = X - mu

cov = np.cov(X)

w, V = np.linalg.eigh(cov)

Xr = V.T @ X_regu

Xw = Xr / np.sqrt(w)[:, None]

plot(X)
plot(X_regu)
plot(Xr)
plot(Xw)
```