import os
import torch
import numpy as np
import lightning.pytorch as pl
from tqdm import tqdm
from scipy.special import softmax
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping
import glob
import re
import logging

# 自定义模块
from dataset import generate_transforms, generate_test_dataloader
from model.model_interface import MInterface
from utils import init_hparams, init_logger, seed_reproducer, load_data

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_best_checkpoints(checkpoints_dir="checkpoints", pattern="fold=*-*.ckpt"):
    """
    自动获取每个fold的最佳模型checkpoint
    
    Args:
        checkpoints_dir: checkpoints文件夹路径，默认为checkpoints/distill
        pattern: 文件名匹配模式
    
    Returns:
        list: 按fold顺序排列的最佳模型路径列表
    """
    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    # 获取所有匹配的checkpoint文件
    all_checkpoints = glob.glob(os.path.join(checkpoints_dir, pattern))
    if not all_checkpoints:
        raise FileNotFoundError(f"No checkpoints found matching pattern: {pattern}")
    
    # 按fold分组
    fold_checkpoints = {}
    for checkpoint in all_checkpoints:
        # 从文件名中提取fold编号和验证指标
        match = re.search(r'fold=(\d+)-.*-(\d+\.\d+)-(\d+\.\d+)\.ckpt', os.path.basename(checkpoint))
        if match:
            fold = int(match.group(1))
            val_loss = float(match.group(2))
            val_acc = float(match.group(3))
            
            if fold not in fold_checkpoints:
                fold_checkpoints[fold] = []
            fold_checkpoints[fold].append((checkpoint, val_loss, val_acc))
    
    # 对每个fold选择最佳模型（验证准确率最高的）
    best_checkpoints = []
    for fold in range(5):  # 假设有5个fold
        if fold not in fold_checkpoints:
            raise ValueError(f"No checkpoints found for fold {fold}")
        
        # 按验证准确率排序
        fold_checkpoints[fold].sort(key=lambda x: x[2], reverse=True)
        best_checkpoint = fold_checkpoints[fold][0][0]
        best_checkpoints.append(best_checkpoint)
        logger.info(f"Selected best checkpoint for fold {fold}: {best_checkpoint}")
    
    return best_checkpoints

def load_checkpoint(model, path, device="cuda"):
    """
    加载 checkpoint 文件，只取 state_dict。
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return model

def main():
    # 固定随机种子
    seed_reproducer(2020)

    # 读取超参数
    hparams = init_hparams()

    # 初始化日志
    logger = init_logger("kun_out", log_dir=hparams.log_dir)

    # 加载数据
    data, test_data = load_data(logger)

    # 加载图像增强
    transforms = generate_transforms(hparams.image_size)

    # 提前定义 EarlyStopping（虽然这里推理用不到）
    early_stop_callback = EarlyStopping(monitor="val_roc_auc", patience=10, mode="max", verbose=True)

    # 初始化模型
    model = MInterface(**vars(hparams))

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=hparams.devices,
        precision="16-mixed" if hparams.precision == 16 else 32,
        logger=False,
        enable_checkpointing=False,
    )

    # 自动获取所有fold的最佳模型
    checkpoint_paths = get_best_checkpoints()
    logger.info(f"Found {len(checkpoint_paths)} best model checkpoints")

    # 生成测试数据加载器
    test_loader = generate_test_dataloader(hparams, test_data, transforms)

    # 多折叠推理并融合
    submission_list = []

    for path in checkpoint_paths:
        model = load_checkpoint(model, path)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Inference {path}"):
                images = batch[0].to(model.device)
                logits = model(images)
                preds.append(logits.cpu())

        preds = torch.cat(preds)
        submission_list.append(softmax(preds.numpy(), axis=1))

    # Ensembling
    submission_ensemble = np.mean(submission_list, axis=0)
    
    # 打印形状信息
    print(f"Test data shape: {test_data.shape}")
    print(f"Submission ensemble shape: {submission_ensemble.shape}")
    print(f"Test data columns: {test_data.columns.tolist()}")

    # 保存最终提交文件
    output_df = test_data.copy()
    # 确保列名匹配
    class_columns = ['healthy', 'multiple_diseases', 'rust', 'scab']
    output_df[class_columns] = submission_ensemble
    output_df.to_csv("data/plant_pathodolgy_data/submission_ensemble.csv", index=False)
    logger.info("Saved ensemble prediction to submission_ensemble.csv.")

if __name__ == "__main__":
    main() 