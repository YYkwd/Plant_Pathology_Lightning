# import os
# import cv2
# import time
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset
# # albumentations 数据增强库
# from albumentations import (
#     Compose, GaussianBlur, HorizontalFlip, MedianBlur, MotionBlur,
#     Normalize, OneOf, RandomBrightnessContrast, Resize, ShiftScaleRotate, VerticalFlip
# )
# from albumentations.pytorch import ToTensorV2 # 转换为 PyTorch Tensor 的接口

# class PlantData(Dataset):
#     def __init__(self, data_csv, soft_labels_filename=None, train=True, image_folder='./images', image_size=(224, 224)):
#         """
#         初始化函数，创建数据集对象时调用。
#         参数说明：
#         - data_csv: 包含图片文件名和标签的CSV路径(必传)
#         - soft_labels_filename: soft label文件(用于自蒸馏,可选)
#         - train: 是否为训练模式（决定是否做数据增强）
#         - image_folder: 存放图片的文件夹路径
#         - image_size: 最终图片resize的目标大小(H, W)
#         """

#         self.data = pd.read_csv(data_csv) # 加载包含图像路径+标签的数据表
#         self.soft_labels = pd.read_csv(soft_labels_filename) if soft_labels_filename else None
#         self.train = train
#         self.image_folder = image_folder
#         self.image_size = image_size

#         self.transforms = self.build_transforms()

#     def build_transforms(self):
#         """
#         根据是训练还是验证模式构建对应的图像预处理 pipeline。
#         使用 Albumentations 组合多个增强操作。
#         """
#         if self.train:
#             return Compose([
#                 Resize(*self.image_size),  # 调整图像到指定大小
#                 OneOf([
#                     RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
#                 ], p=1.0),  # 亮度/对比度扰动

#                 OneOf([
#                     MotionBlur(blur_limit=3, p=1.0),
#                     MedianBlur(blur_limit=3, p=1.0),
#                     GaussianBlur(blur_limit=3, p=1.0)
#                 ], p=0.5),  # 模糊扰动（三选一）

#                 VerticalFlip(p=0.5),  # 垂直翻转
#                 HorizontalFlip(p=0.5),  # 水平翻转
#                 ShiftScaleRotate(  # 平移缩放旋转扰动
#                     shift_limit=0.2, scale_limit=0.2, rotate_limit=20,
#                     interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=1.0
#                 ),
#                 Normalize(  # 图像标准化（用 ImageNet 的均值和方差）
#                     mean=(0.485, 0.456, 0.406),
#                     std=(0.229, 0.224, 0.225),
#                     max_pixel_value=255.0
#                 ),
#                 ToTensorV2()  # 转为 PyTorch Tensor
#             ])
#         else:
#             return Compose([
#                 Resize(*self.image_size),
#                 Normalize(
#                     mean=(0.485, 0.456, 0.406),
#                     std=(0.229, 0.224, 0.225),
#                     max_pixel_value=255.0
#                 ),
#                 ToTensorV2()
#             ])


#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         """
#         返回一个样本，包含：
#         - 图像 Tensor
#         - 标签 Tensor(soft label 或 one-hot label)
#         - 文件名字符串
#         """
#         start_time = time.time()  # 用于记录加载时间（可选统计）

#         row = self.data.iloc[index]  # 获取对应行数据

#         # 获取图像路径
#         image_path = os.path.join(self.image_folder, row[0] + ".jpg")
#         image = cv2.imread(image_path)  # BGR 格式读入
#         if image is None:
#             raise FileNotFoundError(f"Image not found at: {image_path}")

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为 RGB 格式

#         # 如果图像大小不符合预设（有的图像可能尺寸错乱），手动resize
#         if image.shape != (*self.image_size, 3):
#             image = cv2.resize(image, self.image_size)

#         # 应用预处理/增强
#         image = self.transforms(image=image)["image"]

#         # 标签处理（soft label 模式：混合硬标签 + 软标签）
#         if self.soft_labels is not None:
#             label = (
#                 (row[1:].values.astype(np.float32) * 0.7) +
#                 (self.soft_labels.iloc[index, 1:].values.astype(np.float32) * 0.3)
#             )
#         else:
#             label = row[1:].values.astype(np.float32)

#         label = torch.tensor(label, dtype=torch.float32)  # 转为 tensor
#         filename = row[0]  # 提取样本文件名（无后缀）

#         return image, label, filename

