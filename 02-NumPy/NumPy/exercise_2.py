import numpy as np
image=np.random.randint(0,256,(10,10))
print(image.shape)#图片大小
image_1=image.mean()#图片平均亮度
print(image_1)
print(image.max())#最大亮度
print(image.min())#最小亮度
bright_image = np.clip(image + 10, 0, 255).astype(np.uint8)#图片亮度+10，要进行截断，8位灰度图合法范围是（0,255）
print(bright_image)
crop=image[2:7,3:8]#截取图片
print(crop)
#找出亮度大于200的像素
bright_pixels=image[image>200]
print(bright_pixels)
print(bright_pixels.size)
#设函数返回结果
def analyze_image(image):
    mean=image.mean()
    max_image=image.max()
    min_image=image.min()
    bright_pixels=image[image>200]
    num=bright_pixels.size
    return mean,max_image,min_image,num
results=analyze_image(image)
print(results)
    