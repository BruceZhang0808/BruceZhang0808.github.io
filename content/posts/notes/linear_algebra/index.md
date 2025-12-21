---
title: "线性代数学习笔记"
date: '2025-12-21'
draft: false
tags: ["learning", "linear_algebra"]
categories: ["notes"]
author: "Bruce Zhang"
summary: "《线性代数》机械工业出版社，学习笔记"
---

## 线性方程组

### 行化简与阶梯型矩阵，进一步化为简化阶梯形矩阵
- 先导元素：非零行的先导元素是指该行最左边的非零元素
- 阶梯型矩阵、简化阶梯形矩阵![[Pasted image 20250913114233.png]]
![[Pasted image 20250913114250.png]]
- 主元位置：矩阵中的主元位置是$A$中对应于它的阶梯形中先导元素为1的位置。主元列是$A$中含有主元位置的列。
**例题** 求通解，增广矩阵如下
$$\begin{bmatrix} 1 & 6 & 2 &-5 & -2 & -4 \\ 0 & 0 & 2 & -8 & -1 & 3 \\ 0 & 0 & 0 & 0 & 1 & 7 \end{bmatrix}$$
## 矩阵代数

### 矩阵的逆
- 求$A^{-1}$的算法：把增广矩阵 $\begin{bmatrix} A & I \end{bmatrix}$ 进行化简。若 $A$ 行等价与 $I$，则 $\begin{bmatrix} A & I \end{bmatrix}$ 行等价与 $\begin{bmatrix} I & A^{-1} \end{bmatrix}$ ，否则 $A$ 没有逆。
![[Pasted image 20250913120726.png]]
==**注意：仔细理解初等行变换在求解矩阵的逆中发挥作用，且仔细理解上面观点中增广列的优势，后续很多与逆矩阵相关的问题都有所涉及**==
### 维数和秩
- 列空间：矩阵A的列空间是A的各列的线性组合的集合，记作Col A。
- 零空间：矩阵A的零空间是齐次方程 $Ax=0$ 的所有解的集合，记为Nul A。
**例题** 求下列矩阵的零空间的基$$A=\begin{bmatrix} -3 & 6 & -1 & 1 & -7 \\ 1 & -2 & 2 & 3 & -1 \\ 2 & -4 & 5 & 8 & -4 \end{bmatrix}$$
### 矩阵因式分解
![[Pasted image 20250914103841.png]]
#### LU分解
- 定义：$A_{m \times n}=L_{m \times m}U_{m \times n}$  ，其中$L$是下三角矩阵， 主对角线全 为1，$U$是阶梯形矩阵
![[Pasted image 20250913121600.png]]
> 证明：
> 如果可能的话，经过多次行倍加变换可以将矩阵$A$化为阶梯形，这样存在多个单位下三角矩阵$E_p,...E_2,E_1$，使得$E_p...E_2E_1A=U$，即$A=(E_p...E_2E_1)^{-1}U$，由于单位下三角矩阵的乘积和逆矩阵仍为单位下三角矩阵，则$A=LU$
- 适用条件：A可以只用行倍加变换化为阶梯形
- 作用：方便求解线性方程组$Ax=b -> L(Ux)=b -> 令y=Ux -> Ly=b \ , Ux=y$ 
- 计算方法：先将A化为阶梯形矩阵U，然后填充L的元素
**例题** $$
\mathbf{A}_1=
\begin{bmatrix}
2 & 4 & -1 & 5 & -2 \\
-4 & -5 & 3 & -8 & 1 \\
2 & -5 & -4 & 1 & 8 \\
-6 & 0 & 7 & -3 & 1
\end{bmatrix}
\qquad
\mathbf{A}_2=
\begin{bmatrix}
2 & -4 & -2 & 3 \\
6 & -9 & -5 & 8 \\
2 & -7 & -3 & 9 \\
4 & -2 & -2 & -1 \\
-6 & 3 & 3 & 4
\end{bmatrix}
$$

#### 满秩分解
- 定义：$A_{m\times n} = B_{m\times r}C_{r \times n}$ ，其中 $A,B,C$ 的秩均为 $r$ 
- 计算方法：仅通过初等行变换即可实现。先对 $A$ 进行行初等变换为简化阶梯形 $C'$ ，按照 $A$ 的主元列顺序组成满列秩矩阵 $B$ 的列向量，然后按 $C'$ 的非零行顺序组成满行秩矩阵 $C$ 。最后满秩分解为 $A=BC$ 。
**例题**
$$
\begin{aligned}
A_1 =
\begin{bmatrix}
1 & 4 & -1 & 5 & 6 \\
2 & 0 & 0 & 0 & -14 \\
-1 & 2 & -4 & 0 & 1 \\
2 & 6 & -5 & 5 & -7
\end{bmatrix}
\qquad
A_2 =
\begin{bmatrix}
1 & 3 & 2 & 1 & 4 \\
2 & 6 & 1 & 0 & 7 \\
3 & 9 & 3 & 1 & 11
\end{bmatrix}
\end{aligned}
$$
#### 特征值分解
 - 定义：将矩阵A分解为其特征值和特征向量的乘积，即 $A=P\Lambda P^{-1}$ ，其中 $\Lambda$ 为 $diag\{\lambda_1\ \lambda_2\ ... \ \lambda_n\}$ ，$P$ 为每个特征值所对应的特征向量为列向量构成的矩阵。 
 > 证明：
 > 对于求出的A的每一个特征值$\lambda_i$对应的特征向量$x_i$，将其作为列向量拼成矩阵$X$，合并$Ax=\lambda x$，得到$AX=X\Lambda$，即$A=X\Lambda X^{-1}$
- 适用条件：A可以对角化，用于方阵
- 应用：PCA，特征提取，······
- 对特征值分解再进一步，得到谱分解
#### 谱分解
详见[[#对称矩阵和谱定理]]
- 一般的可对角化的矩阵：将 $P$ 用它的列向量表示，$P^{-1}$ 用行向量表示，得
$$ 
\begin{equation}
\begin{aligned}
A &=Pdiag(\lambda_1,\lambda_2,\cdots,\lambda_n)P^{-1} \\
&= \begin{bmatrix} \alpha_1 & \alpha_2 & \cdots & \alpha_n \end{bmatrix} 
\begin{bmatrix}\lambda_1 \\  &  \lambda_2 \\ & & \ddots \\ & & & \lambda_n\end{bmatrix}
\begin{bmatrix}\beta_1^T \\ \beta_2^T \\ \vdots \\ \beta_n^T \end{bmatrix}\\
&=\lambda_1\alpha_1\beta_1^T+\lambda_2\alpha_2\beta_2^T+\cdots+\lambda_n\alpha_n\beta_n^T
\end{aligned}
\end{equation}
$$
- 对称矩阵：如果A是对称矩阵，则P的列是A的单位正交特征向量 $u_1, u_2,\cdots, u_n$ ，且由于 $P^{-1}=P^T$ ，故
$$
\begin{equation}
\begin{aligned}
A &=Pdiag(\lambda_1,\lambda_2,\cdots,\lambda_n)P^{T} \\
&= \begin{bmatrix} u_1 & u_2 & \cdots & u_n \end{bmatrix} 
\begin{bmatrix}\lambda_1 \\  &  \lambda_2 \\ & & \ddots \\ & & & \lambda_n\end{bmatrix}
\begin{bmatrix}u_1^T \\ u_2^T \\ \vdots \\ u_n^T \end{bmatrix}\\
&=\lambda_1u_1u_1^T+\lambda_2u_2u_2^T+\cdots+\lambda_nu_nu_n^T
\end{aligned}
\end{equation}
$$
**这样就将矩阵A分解为了由A的谱（特征值）所确定的小块，故名谱分解**
#### QR分解
![[Pasted image 20250914102234.png]]
> 证明：
> $A$ 可以写为 $A=\begin{bmatrix}\alpha_1 & \alpha_2 & ... & \alpha_n \end{bmatrix}$，将$A$的列向量进行施密特正交化变换为$Col\ A$的标准正交基，构成矩阵 $Q=\begin{bmatrix}q_1 & q_2 & ... & q_n \end{bmatrix}$ ，每个 $q_i$ 在$\alpha$下的坐标作为列构成上三角矩阵 $R$ ，且主对角线元素均为正数
#### 奇异值分解
详见[[#奇异值和奇异值分解]]
![[Pasted image 20250914172854.png]]

## 行列式
描述的是线性空间中基底向量所围立体的体积，也看做是变换后图形的面积或体积扩大、缩小了多少倍

## 向量空间
### 基的变换
![[Pasted image 20250913163708.png]]
其中$\mathop{P}\limits_{C \leftarrow B}$ 称为**由$B$到$C$的坐标变换矩阵**，注意，这里是坐标变换，非**基变换**。
仔细比较如下两个式子，此时P也称为C到B的过渡矩阵
$$
\mathop{P}\limits_{C \leftarrow B}\begin{bmatrix}x\end{bmatrix}_B=\begin{bmatrix}x\end{bmatrix}_C
\qquad B=C\mathop{P}\limits_{C \leftarrow B}
$$
**例题**
![[Pasted image 20250913165910.png]]

## 特征值和特征向量
![[Pasted image 20250913182722.png]]
![[Pasted image 20250913182743.png]]
以上构成特征值分解 $A=P\Lambda P^{-1}$ ，其中 $\Lambda$ 为 $diag\{\lambda_1\ \lambda_2\ ... \ \lambda_n\}$ ，$P$ 为每个特征值所对应的特征向量为列向量构成的矩阵。 

## 正交性和最小二乘法
### 正交性
- 正交补![[Pasted image 20250914113434.png]]
- 正交投影![[Pasted image 20250914113732.png]]
- Gram-Schmidt正交化方法![[Pasted image 20250914113912.png]]
### 最小二乘法
- 方法提出：若方程组 $Ax=b$ 的解不存在，但又需要求解时，最好的办法就是寻找 $x$，使得 $Ax$ 尽可能接近$b$ 一般的最小二乘问题就是找到使 $||b-Ax||$ 最小的 $x$ ![[Pasted image 20250914114409.png]]
- 一般最小二乘问题解法：因为有 $b-A\hat{x}$ 与 $Col\ A$ 正交，故 $A^{T}(b-A\hat{x})=0$ ，故有 $\boldsymbol{A^TA\hat{x}=A^Tb}$ ，这表明 $Ax=b$ 的每个最小二乘解都满足方程 ${A^TAx=A^Tb}$ ，该方程也被称为 $Ax=b$ 的法方程，其解通常用 $\hat{x}$ 表示。求解法方程的非空解集，即是最小二乘解。
- 若 $A$ 的列向量彼此正交：直接写出 $b$ 在 $Col\ A$ 上的正交投影，即利用 $A$ 的列线性表示 $b$ ，系数即为 $\hat{x}$ 
- 若 $A$ 的列线性无关，最小二乘解可通过A的QR分解更可靠的求出![[Pasted image 20250914120955.png]]

## 对称矩阵和二次型
### 对称矩阵和谱定理
- 对称矩阵的特征向量彼此正交，通过Gram-Schmidt方法可以变为标准正交基，且可以正交对角化
- **谱定理**：矩阵A的特征值的集合被称为A的*谱*，下面关于特征值的描述被称为*谱定理*![[Pasted image 20250914154122.png]]
- 谱分解：![[Pasted image 20250914155947.png]]上的正交投影
### 奇异值和奇异值分解
- 奇异值：![[Pasted image 20250914171926.png]]

- 奇异值分解（Singular value decomposition，SVD）![[Pasted image 20250914172356.png]]![[Pasted image 20250914172257.png]]
