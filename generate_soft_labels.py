import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
from scipy.special import softmax
from torch.utils.data import DataLoader
import logging

# 导入自定义模块
from dataset import generate_transforms, PlantDataset
from model.model_interface import MInterface
from utils import init_hparams, seed_reproducer, load_data, IMAGE_FOLDER

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model, path, device="cuda"):
    """加载模型权重"""
    try:
        logger.info(f"Loading model from {path}")
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model from {path}: {str(e)}")
        raise

def generate_soft_labels(hparams, data, transforms, checkpoint_paths):
    """
    使用训练好的Fold模型，在验证集上生成Soft Labels，并与硬标签合并。
    
    Args:
        hparams: 超参数
        data: 训练数据DataFrame，包含硬标签（healthy, multiple_diseases, rust, scab）
        transforms: 数据增强转换
        checkpoint_paths: 模型权重路径列表
    """
    # 检查数据目录
    if not os.path.exists(IMAGE_FOLDER):
        raise ValueError(f"Image folder not found: {IMAGE_FOLDER}")
    
    # 检查数据列
    label_columns = ['healthy', 'multiple_diseases', 'rust', 'scab']
    if not all(col in data.columns for col in label_columns):
        raise ValueError(f"Data must contain label columns: {label_columns}")
    
    folds = KFold(n_splits=5, shuffle=True, random_state=hparams.seed)
    soft_labels_dfs = []
    alpha = 0.7  # 硬标签权重
    beta = 0.3   # 软标签权重

    for fold_idx, (train_idx, val_idx) in enumerate(folds.split(data)):
        logger.info(f"\nProcessing Fold {fold_idx + 1}/5...")
        
        # 准备验证集数据
        val_data = data.iloc[val_idx, :].reset_index(drop=True)
        logger.info(f"Validation set size: {len(val_data)}")
        
        # 提取硬标签（从第2列开始，跳过image_id列）
        hard_labels = val_data[label_columns].values.astype(np.float32)
        logger.info(f"Hard labels shape: {hard_labels.shape}")
        logger.info(f"Hard labels sample:\n{hard_labels[:5]}")  # 显示前5个样本的硬标签
        
        val_dataset = PlantDataset(
            data=val_data,
            transforms=transforms["val_transforms"]
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=hparams.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # 加载当前fold的模型
        model = MInterface(**vars(hparams))
        model = load_model(model, checkpoint_paths[fold_idx])

        # 生成预测（软标签）
        preds = []
        with torch.no_grad():
            for images, labels, times in tqdm(val_dataloader, desc=f"Fold {fold_idx} Soft Labels"):
                try:
                    images = images.to(model.device)
                    outputs = model(images)
                    preds.append(outputs.cpu())
                except Exception as e:
                    logger.error(f"Error during prediction: {str(e)}")
                    raise

        # 合并预测结果并转换为概率
        preds = torch.cat(preds)
        soft_probs = softmax(preds.numpy(), axis=1)
        logger.info(f"Soft labels shape: {soft_probs.shape}")
        logger.info(f"Soft labels sample:\n{soft_probs[:5]}")  # 显示前5个样本的软标签
        
        # 合并软标签和硬标签
        mixed_labels = alpha * hard_labels + beta * soft_probs
        logger.info(f"Mixed labels shape: {mixed_labels.shape}")
        logger.info(f"Mixed labels sample:\n{mixed_labels[:5]}")  # 显示前5个样本的混合标签
        
        # 保存当前fold的混合标签
        val_data_copy = val_data.copy()
        val_data_copy[label_columns] = mixed_labels  # 使用列名更新标签
        soft_labels_dfs.append(val_data_copy)
        logger.info(f"Completed Fold {fold_idx + 1}")

    # 合并所有fold的标签
    mixed_labels_df = data[["image_id"]].merge(pd.concat(soft_labels_dfs), how="left", on="image_id")
    
    # 保存混合标签
    output_path = os.path.join("data", "plant_pathodolgy_data", "images",  "mixed_labels.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mixed_labels_df.to_csv(output_path, index=False)
    logger.info(f"\n✅ Mixed labels (hard + soft) saved to {output_path}")
    
    # 同时保存纯软标签（可选）
    soft_labels_df = data[["image_id"]].merge(pd.concat(soft_labels_dfs), how="left", on="image_id")
    soft_output_path = os.path.join("data", "images", "plant_pathodolgy_data", "soft_labels.csv")
    soft_labels_df.to_csv(soft_output_path, index=False)
    logger.info(f"✅ Pure soft labels saved to {soft_output_path}")

def main():
    try:
        # 固定随机种子
        seed_reproducer(2020)
        logger.info("Random seed set to 2020")
        
        # 初始化超参数
        hparams = init_hparams()
        logger.info("Hyperparameters initialized")
        
        # 加载数据
        data, _ = load_data(None)
        logger.info(f"Data loaded, shape: {data.shape}")
        
        # 确保数据包含正确的列
        required_columns = ['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # 图像增强
        transforms = generate_transforms(hparams.image_size)
        logger.info("Transforms generated")
        
        # 训练好的5折模型路径 - 使用checkpoints目录中的最佳模型
        checkpoint_paths = [
            "checkpoints/fold=0-74-0.2447-0.9588.ckpt",  # fold 0 最佳模型
            "checkpoints/fold=1-63-0.1451-0.9704.ckpt",  # fold 1 最佳模型
            "checkpoints/fold=2-94-0.2018-0.9586.ckpt",  # fold 2 最佳模型
            "checkpoints/fold=3-73-0.2700-0.9592.ckpt",  # fold 3 最佳模型
            "checkpoints/fold=4-60-0.1950-0.9851.ckpt",  # fold 4 最佳模型
        ]
        
        # 检查模型文件是否存在
        for path in checkpoint_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model checkpoint not found: {path}")
        
        # 生成软标签
        generate_soft_labels(hparams, data, transforms, checkpoint_paths)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 