---
title: "latex 学习笔记"
date: '2025-12-10'
draft: false
tags: ["latex", "learning"]
categories: ["notes"]
author: "Bruce Zhang"

summary: "持续更新 latex 学习笔记"
---

## 美观的表格写法

推荐网站 [Tables Generator](https://tablesgenerator.com/)，先把表格内容填进去，再选样式为 Booktabs table style，还可以调整 Caption 和 Layout。

## 插入图片的方法

1. 更像表格排版，子图无小标题和编号
```tex
% \usepackage{graphicx}
% \usepackage{float}  想用 [H] 就加
\begin{figure}[H]   % 用H表示固定在此不乱动
\centering
\setlength{\tabcolsep}{3pt} % 列间距
\renewcommand{\arraystretch}{1.0}

\begin{tabular}{ccc}
\includegraphics[width=0.32\textwidth]{AI-assignment/figures/GA/20个城市收敛曲线.pdf} &
\includegraphics[width=0.32\textwidth]{AI-assignment/figures/PSO/20个城市收敛图.pdf} &
\includegraphics[width=0.32\textwidth]{AI-assignment/figures/ACO/20个城市收敛曲线.pdf} \\

\includegraphics[width=0.32\textwidth]{AI-assignment/figures/GA/20个城市最佳路径图.pdf} &
\includegraphics[width=0.32\textwidth]{AI-assignment/figures/PSO/20个城市路径图.pdf} &
\includegraphics[width=0.32\textwidth]{AI-assignment/figures/ACO/20个城市最佳路径图.pdf} \\
\end{tabular}

\caption{GA、PSO 与 ACO 的收敛曲线（上）与路径图（下）对比}
\label{fig:compare-2x3}
\end{figure}
```

2. 每个子图都有小标题，用`subcaption`
```tex
% \usepackage{graphicx}
% \usepackage{subcaption}
% \usepackage{float}

\begin{figure}[H]
\centering

% ---------- 第 1 行：收敛曲线 ----------
\begin{subfigure}{0.32\textwidth}
  \centering
  \includegraphics[width=\linewidth]{AI-assignment/figures/GA/20个城市收敛曲线.pdf}
  \caption{GA 收敛曲线（20 城市）}
\end{subfigure}\hfill
\begin{subfigure}{0.32\textwidth}
  \centering
  \includegraphics[width=\linewidth]{AI-assignment/figures/PSO/20个城市收敛图.pdf}
  \caption{PSO 收敛曲线（20 城市）}
\end{subfigure}\hfill
\begin{subfigure}{0.32\textwidth}
  \centering
  \includegraphics[width=\linewidth]{AI-assignment/figures/ACO/20个城市收敛曲线.pdf}
  \caption{ACO 收敛曲线（20 城市）}
\end{subfigure}

\vspace{0.6em}

% ---------- 第 2 行：路径图 ----------
\begin{subfigure}{0.32\textwidth}
  \centering
  \includegraphics[width=\linewidth]{AI-assignment/figures/GA/20个城市最佳路径图.pdf}
  \caption{GA 最佳路径（20 城市）}
\end{subfigure}\hfill
\begin{subfigure}{0.32\textwidth}
  \centering
  \includegraphics[width=\linewidth]{AI-assignment/figures/PSO/20个城市路径图.pdf}
  \caption{PSO 路径图（20 城市）}
\end{subfigure}\hfill
\begin{subfigure}{0.32\textwidth}
  \centering
  \includegraphics[width=\linewidth]{AI-assignment/figures/ACO/20个城市最佳路径图.pdf}
  \caption{ACO 最佳路径（20 城市）}
\end{subfigure}

\caption{GA、PSO 与 ACO 的收敛曲线（上）与路径图（下）对比}
\label{fig:compare-2x3}
\end{figure}
```
