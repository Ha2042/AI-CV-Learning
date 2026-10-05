import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 动态获取路径并读取图片
base_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(base_dir, 'img_1.jpg')
img_hwc = plt.imread(img_path)

print("原图形状 (H, W, C):", img_hwc.shape) # 比如 (1456, 732, 3)

# 2. 转换为深度学习标准格式 (C, H, W)
img_chw = np.transpose(img_hwc, axes=(2, 0, 1))
print("转换后形状 (C, H, W):", img_chw.shape)

# 3. 增加一个批次维度 B，变成 (B, C, H, W)
img_bchw = img_chw[np.newaxis, :, :, :]
print("添加批次后 (B, C, H, W):", img_bchw.shape)

# ================== 开始画图：见证奇迹的时刻 ==================
# 设置中文字体（防止标题乱码，Windows 默认支持 SimHei）
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(16, 10))

# 图 1：正常显示原图 (H, W, C)
plt.subplot(2, 2, 1)
plt.imshow(img_hwc)
plt.title(f"1. 原图 (H, W, C) {img_hwc.shape}")
plt.axis('off')

# 图 2：转置 (C, H, W) 后，直接把第 0 个通道（红色通道）画出来
# 当图片变成 (C, H, W) 后，img_chw[0] 就是 (H, W) 的二维数据，可以直接 imshow 显示灰度
plt.subplot(2, 2, 2)
plt.imshow(img_chw[0], cmap='Reds') # 用红色色彩映射，凸显这是红色通道
plt.title(f"2. 提取(C,H,W)的C=0通道 {img_chw[0].shape}")
plt.axis('off')

# 图 3：见证 transpose 的几何威力！交换 H 和 W (轴0和轴1)
# 注意：这里我们交换 H 和 W，图片会“躺下”
img_swap = np.transpose(img_hwc, axes=(1, 0, 2))
plt.subplot(2, 2, 3)
plt.imshow(img_swap)
plt.title(f"3. 交换 H 和 W 轴 {img_swap.shape} (图片躺平了)")
plt.axis('off')

# 图 4：把加了批次 (B, C, H, W) 的数据还原成人类能看的格式
# 步骤：去掉 B 维度(img_bchw[0]) -> 变回 (C, H, W) -> 转置回 (H, W, C) -> 显示
img_show_bchw = np.transpose(img_bchw[0], axes=(1, 2, 0))
plt.subplot(2, 2, 4)
plt.imshow(img_show_bchw)
plt.title(f"4. (B,C,H,W) 还原展示 {img_show_bchw.shape}")
plt.axis('off')

plt.tight_layout()
plt.show()