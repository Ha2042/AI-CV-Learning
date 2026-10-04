# 05-MLP+CNN

# 2026-10-04

# Today I learned

- 卷积层
- 最大池化应用
- 非线性激活
- 线性层

# Problem

## 模型模式切换错误

- 错误写法：model.train(images)
- 正确写法：model.train()  output=model(images)

## 如何提高Accuracy

- 增加epoch(注意过拟合)
- 增加卷积层/池化层（注意特征图尺寸和计算量）
- 换优化器（SGD->Adam）
- 数据增强

# Next

Grasp the big picture of computer vision tasks like classification, detection, segmentation, and saliency.

