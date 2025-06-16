import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from albumentations import (
    Compose, GaussianBlur, HorizontalFlip, MedianBlur, MotionBlur,
    Normalize, OneOf, RandomBrightnessContrast, Resize, ShiftScaleRotate, VerticalFlip
)
from utils import IMAGE_FOLDER, IMG_SHAPE

# Dataset for Plant Pathology Challenge
class PlantDataset(Dataset):
    def __init__(self, data, soft_labels_filename=None, transforms=None):
        self.data = data
        self.transforms = transforms
        if soft_labels_filename:
            self.soft_labels = pd.read_csv(soft_labels_filename)
        else:
            self.soft_labels = None
            print("Soft labels not provided. Using original labels.")

    def __getitem__(self, index):
        start_time = time.time()

        # Get image_id
        image_id = self.data.iloc[index, 0]

        # Read image
        image_path = os.path.join(IMAGE_FOLDER, image_id + ".jpg")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # H W C

        # Correct image shape if necessary
        if image.shape != IMG_SHAPE:
            image = image.transpose(1, 0, 2) #高度和宽度搞反了

        # Apply transformations
        if self.transforms:
            image = self.transforms(image=image)["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        # Prepare labels
        if self.soft_labels is not None:
            label = (
                (self.data.iloc[index, 1:].values.astype(np.float32) * 0.7) +
                (self.soft_labels.iloc[index, 1:].values.astype(np.float32) * 0.3)
            )
        else:
            label = self.data.iloc[index, 1:].values.astype(np.float32)

        label = torch.tensor(label, dtype=torch.float32)
        data_load_time = time.time() - start_time

        return image, label, data_load_time, image_id

    def __len__(self):
        return len(self.data)


def generate_transforms(image_size):
    """Generate albumentations transforms for training and validation."""
    train_transform = Compose([
        Resize(height=image_size[0], width=image_size[1]),
        OneOf([
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
        ], p=1.0),
        OneOf([
            MotionBlur(blur_limit=3, p=1.0),
            MedianBlur(blur_limit=3, p=1.0),
            GaussianBlur(blur_limit=3, p=1.0)
        ], p=0.5),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.2, scale_limit=0.2, rotate_limit=20,
            interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=1.0
        ),
        Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0
        ),
    ])

    val_transform = Compose([
        Resize(height=image_size[0], width=image_size[1]),
        Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0
        ),
    ])

    return {"train_transforms": train_transform, "val_transforms": val_transform}


def generate_dataloaders(hparams, train_data, val_data, transforms):
    """
    生成训练和验证数据加载器
    
    Args:
        hparams: 超参数
        train_data: 训练数据集
        val_data: 验证数据集
        transforms: 数据增强转换
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    # 创建训练和验证数据集
    train_dataset = PlantDataset(
        data=train_data,
        transforms=transforms["train_transforms"],
        soft_labels_filename=hparams.soft_labels_filename
    )
    val_dataset = PlantDataset(
        data=val_data,
        transforms=transforms["val_transforms"],
        soft_labels_filename=hparams.soft_labels_filename
    )

    # 训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,  # 使用统一的batch_size
        shuffle=True,
        num_workers=hparams.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=True
    )
    
    # 验证数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=hparams.batch_size,  # 使用统一的batch_size
        shuffle=False,
        num_workers=hparams.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=True
    )
    
    return train_loader, val_loader

def generate_test_dataloader(hparams, test_data, transforms):
    """
    生成测试数据加载器
    
    Args:
        hparams: 超参数
        test_data: 测试数据集
        transforms: 数据增强转换
    
    Returns:
        test_loader: 测试数据加载器
    """
    # 创建测试数据集
    test_dataset = PlantDataset(
        data=test_data,
        transforms=transforms["val_transforms"],  # 使用验证集的转换
        soft_labels_filename=None  # 测试集通常没有软标签
    )
    
    # 测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=hparams.batch_size,  # 使用统一的batch_size
        shuffle=False,
        num_workers=hparams.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=True
    )
    
    return test_loader
