import os
os.environ["A_SKIP_CHECK"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import gc
from time import time
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch
from dataset import generate_transforms, generate_dataloaders, generate_test_dataloader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from model.model_interface import MInterface
from utils import init_hparams, init_logger, print_hparams, seed_reproducer, load_data 

if __name__ == "__main__":
    seed_reproducer(2020)
    hparams = init_hparams()

    #检查超参数 按 key 排序输出
    print_hparams(hparams, mode="table")

    
    torch.set_float32_matmul_precision('medium') # ❗

    logger = init_logger("kun_out", log_dir=hparams.log_dir)
    data, test_data = load_data(logger)
    transforms = generate_transforms(hparams.image_size)

    valid_roc_auc_scores = []
    folds = KFold(n_splits=5, shuffle=True, random_state=hparams.seed)

    for fold_i, (train_index, val_index) in enumerate(folds.split(data)):
        train_data = data.iloc[train_index, :].reset_index(drop=True)
        val_data = data.iloc[val_index, :].reset_index(drop=True)

        train_loader, val_loader = generate_dataloaders(hparams, train_data, val_data, transforms) # 把 train_data, val_data 分别封装成 train_loader 和 val_loader

        #配置模型保存策略（checkpoint）
        checkpoint_callback = ModelCheckpoint(
            monitor="val_roc_auc", #self.log("val_roc_auc", ...) 中使用了 on_epoch=True 实时监控
            save_top_k=1,
            mode="max",
            filename=f"fold={fold_i}" + "-{epoch}-{val_loss:.4f}-{val_roc_auc:.4f}",
            auto_insert_metric_name=False,
        )
        #配置早停策略（early stopping）
        early_stop_callback = EarlyStopping(monitor="val_f1", patience=10, mode="max")

        model = MInterface(**vars(hparams))
        print(type(model))

        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices= hparams.devices,
            min_epochs=100,
            max_epochs=hparams.max_epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
            precision="16-mixed" if hparams.precision == 16 else 32,
            enable_progress_bar=True, #开启 tqdm 动态进度条
            enable_model_summary=False,
            logger=False, 
            num_sanity_val_steps=0,
        )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        valid_roc_auc_scores.append(round(checkpoint_callback.best_model_score.item(), 4))
        logger.info(valid_roc_auc_scores)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # # 在所有fold训练完成后进行测试
    # logger.info("Starting final testing...")
    
    # # 加载测试数据
    # test_loader = generate_test_dataloader(hparams, test_data, transforms)
    
    # # 加载最佳模型进行测试
    # best_model_path = checkpoint_callback.best_model_path
    # logger.info(f"Loading best model from: {best_model_path}")
    
    # # 创建新模型实例并加载权重
    # test_model = MInterface(**vars(hparams))
    # test_model.load_state_dict(torch.load(best_model_path)['state_dict'])
    
    # # 设置模型为评估模式
    # test_model.eval()
    
    # # 创建测试trainer
    # test_trainer = pl.Trainer(
    #     accelerator="gpu" if torch.cuda.is_available() else "cpu",
    #     devices=hparams.devices,
    #     logger=False,
    #     enable_progress_bar=True
    # )
    
    # # 运行测试
    # test_results = test_trainer.test(test_model, dataloaders=test_loader)
    
    # # 记录测试结果
    # logger.info("Test Results:")
    # logger.info(f"Test Loss: {test_results[0]['test_loss']:.4f}")
    # logger.info(f"Test ROC AUC: {test_results[0]['test_roc_auc']:.4f}")
    
    # # 清理
    # del test_model
    # gc.collect()
    # torch.cuda.empty_cache()