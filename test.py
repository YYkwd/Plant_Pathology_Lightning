import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["A_SKIP_CHECK"] = "1"
import gc
import lightning.pytorch as pl
import torch
from dataset import generate_transforms, generate_test_dataloader
from model.model_interface import MInterface
from utils import init_hparams, init_logger, seed_reproducer, load_data

def test_model(model_path, hparams):
    """
    使用训练好的模型进行测试
    
    Args:
        model_path: 模型权重文件路径
        hparams: 超参数
    """
    # 设置随机种子
    seed_reproducer(2020)
    
    # 设置矩阵乘法精度
    torch.set_float32_matmul_precision('medium')
    
    # 初始化日志
    logger = init_logger("test_out", log_dir=hparams.log_dir)
    
    # 加载测试数据
    _, test_data = load_data(logger)
    transforms = generate_transforms(hparams.image_size)
    test_loader = generate_test_dataloader(hparams, test_data, transforms)
    
    # 创建模型实例并加载权重
    test_model = MInterface(**vars(hparams))
    test_model.load_state_dict(torch.load(model_path, weights_only=True)['state_dict'])
    test_model.eval()
    
    # 创建测试trainer
    test_trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=hparams.devices,
        logger=False,
        enable_progress_bar=True
    )
    
    # 运行测试
    test_trainer.test(test_model, dataloaders=test_loader)
    
    # 清理
    del test_model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 初始化超参数
    hparams = init_hparams()
    
    # 指定要测试的模型路径
    model_path = "checkpoints/fold=4-60-0.1950-0.9851.ckpt"  # 使用最佳模型
    
    # 运行测试
    test_model(model_path, hparams) 