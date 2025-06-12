---
title: "16 周深度学习与机器学习全栈学习计划"
date: 2025-06-12
tags:
  - 学习计划
  - 深度学习
  - 机器学习
description: "从线性代数到预训练模型，16 周每天 1.5h 的实战与理论深度学习路线图。"
---

## 📅 16 周学习总览

下面是为期 16 周的学习计划，每周聚焦一个核心主题，配合理论和实战任务，并给出主要参考资料及链接。

| 周次    | 重点主题                        | 任务                                                                                              | 所需资料 / 链接                                                                                                                                     |
| ------- | ------------------------------- | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 第 1 周 | 线性代数基础                    | - 每日 1.5 h：观看 MIT OCW 线性代数 1–2 章视频<br/>- 实战：用 NumPy 实现矩阵加法、乘法             | - [Strang 线性代数 Lecture 1–2 (MIT OCW)](https://ocw.mit.edu)\n- [NumPy 官方教程：数组运算](https://numpy.org/doc/)                                   |
| 第 2 周 | 线性代数进阶                    | - 每日 1.5 h：SVD 与特征分解概念学习<br/>- 实战：用 NumPy 实现 SVD，验证奇异值分解重构效果         | - Strang 教材第 4 章（SVD）<br/>- 《深度学习》（Goodfellow）第 2 章                                                                                 |
| 第 3 周 | 概率论与统计                    | - 每日 1.5 h：高斯分布、伯努利分布、期望与方差<br/>- 练习：计算不同分布下的概率密度与样本生成     | - [Stanford CS109 Lecture Notes](http://stat110.net)<br/>- [Khan Academy 概率论模块](https://www.khanacademy.org/math/statistics-probability)              |
| 第 4 周 | 优化算法入门                    | - 学习梯度下降、学习率、收敛速度<br/>- 实战：用纯 Python 实现梯度下降求解线性回归                   | - 《深度学习》Adam / SGD 章节<br/>- Andrew Ng 机器学习 Week 1–2 编程作业（Octave 模板）                                                              |
| 第 5 周 | 监督学习核心—回归与分类        | - 观看 Andrew Ng ML Week 2–3（线性回归）<br/>- 编程：Octave 实现多变量线性回归                     | - [Coursera Andrew Ng ML Course](https://www.coursera.org/learn/machine-learning)<br/>- Octave / MATLAB 安装与入门                                     |
| 第 6 周 | 监督学习核心—逻辑回归与正则化  | - 观看 Andrew Ng ML Week 4–5<br/>- 编程：实现逻辑回归 + L2 正则化                                   | - Andrew Ng ML 作业模板<br/>- 《Pattern Recognition and Machine Learning》（Bishop）相关章节                                                          |
| 第 7 周 | 经典模型扩展—SVM 与核方法      | - 学习 SVM 原理与核技巧<br/>- 练习：用 scikit-learn 训练不同核函数的 SVM，比较效果               | - Coursera ML Week 8 视频<br/>- [scikit-learn 文档（SVC、核函数）](https://scikit-learn.org/stable/modules/svm.html)                                       |
| 第 8 周 | 无监督学习—聚类与降维          | - 复习 PCA 原理并用 sklearn 实现<br/>- 学习 k-means，实战：在 MNIST 子集上做聚类                   | - Andrew Ng ML Week 9–10<br/>- [scikit-learn 聚类与 PCA 教程](https://scikit-learn.org/stable/modules/clustering.html)                                     |
| 第 9 周 | 深度学习入门—神经网络基础      | - 学习前向/反向传播算法<br/>- 编程：用纯 Python 实现一个两层神经网络                             | - Coursera Deep Learning Specialization: Course 1 Week 1–2<br/>- 《神经网络与深度学习》（邱锡鹏）第 1 章                                                     |
| 第 10 周| 深度学习框架实战—PyTorch 基础   | - 环境搭建：安装 PyTorch<br/>- 实战：用 PyTorch 实现 MNIST 数据集分类                              | - 官方 [PyTorch 入门教程](https://pytorch.org/tutorials/)<br/>- MNIST 数据集简介与加载文档                                                               |
| 第 11 周| 卷积神经网络—理论与实践        | - 学习卷积原理、池化、常见架构<br/>- 实战：在 CIFAR-10 上复现一个简单的 CNN                        | - CS231n Lecture 7–9 PPT<br/>- PyTorch CNN 示例代码                                                                                                   |
| 第 12 周| RNN 与注意力机制               | - 学习 RNN/LSTM 基本原理<br/>- 编程：用 PyTorch 实现字符级语言模型（简单 RNN）                    | - 《深度学习》（Goodfellow）RNN 章节<br/>- PyTorch 官方 RNN 示例                                                                                     |
| 第 13 周| Transformer 原理               | - 阅读 “Attention Is All You Need” 论文<br/>- 笔记：画出模型架构图                                 | - [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)<br/>- Jay Alammar “The Illustrated Transformer” 博文                                                 |
| 第 14 周| Transformer 实现               | - 实战：参照 Hugging Face 教程，用 PyTorch Lightning 实现简化版 Transformer 编码器               | - Hugging Face “Tutorial: Build a Transformer from scratch”<br/>- PyTorch Lightning 文档                                                               |
| 第 15 周| 预训练模型与微调—BERT          | - 学习 BERT 预训练目标（MLM）、架构差异<br/>- 实战：用 Hugging Face Transformers 微调 BERT        | - Devlin et al. “BERT: Pre-training of Deep Bidirectional Transformers”<br/>- 🤗 [Transformers 文档](https://huggingface.co/docs/transformers)           |
| 第 16 周| 自回归模型—GPT 与文本生成      | - 学习 GPT 系列架构与自回归目标<br/>- 实战：用 Transformers API 生成文本并调整温度、Top-k、Top-p  | - Radford et al. GPT-2 论文<br/>- 🤗 [Transformers “Text Generation” 示例](https://huggingface.co/docs/transformers/task_summary#text-generation)       |

> **备注**：  
> - 建议每天保持 1.5 小时的专注学习；  
> - 实战示例均可结合 Jupyter Notebook 记录过程；  
> - Links 均可直接点击跳转至在线资源。

---
