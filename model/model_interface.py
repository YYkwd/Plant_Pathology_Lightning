import inspect
from sklearn.metrics import roc_auc_score
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import lightning.pytorch as pl
from model.losses.factory import LossFactory
import pandas as pd
import numpy as np
import os
from datetime import datetime

class MInterface(pl.LightningModule):
    def __init__(self,  **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []  # 添加测试输出存储
        self.test_predictions = []  # 存储测试预测结果
        self.test_image_ids = []    # 存储测试图片ID
        
    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        images, labels, data_load_time, image_ids = batch  # 正确解包所有值
        scores = self(images)
        loss = self.loss_function(scores, labels)
        self.training_step_outputs.append(loss.detach())
        return loss
    
    def on_train_epoch_end(self ): #每个训练 epoch 结束时自动调用
        if self.training_step_outputs:
            train_loss_mean = torch.stack(self.training_step_outputs).mean()
            self.log("train_loss", train_loss_mean, prog_bar=True)
            self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        images, labels, data_load_time, image_ids = batch  # 正确解包所有值
        scores = self(images)
        loss = self.loss_function(scores, labels)
        #计算ACC
        label_digit = labels.argmax(dim=1)
        out_digit = scores.argmax(dim=1)

        correct_num = (label_digit == out_digit).sum().item()
        total_num = len(out_digit) #就是当前 batch 的样本总数

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', correct_num/total_num,
                 on_step=False, on_epoch=True, prog_bar=True)
        output = {
            "val_loss": loss.detach(),
            "scores": scores,
            "labels": labels,
            "data_load_time": torch.sum(data_load_time),
            "val_acc" : correct_num / total_num ,
        }
        self.validation_step_outputs.append(output)
        
        return output
    
    def on_validation_epoch_end(self):
        val_loss_mean = torch.stack([o["val_loss"] for o in self.validation_step_outputs]).mean()
        scores_all = torch.cat([o["scores"] for o in self.validation_step_outputs]).cpu()
        labels_all = torch.round(torch.cat([o["labels"] for o in self.validation_step_outputs]).cpu())
        val_roc_auc = roc_auc_score(labels_all, scores_all)

        #self.log("val_loss", val_loss_mean, prog_bar=True)
        self.log("val_roc_auc", val_roc_auc, prog_bar=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        images, _, _, image_ids = batch  # 忽略标签
        scores = self(images)
        
        # 存储预测结果和图片ID
        self.test_predictions.append(scores.detach().cpu())
        self.test_image_ids.extend(image_ids)
        
        return None
    
    def on_test_epoch_end(self):
        if self.test_predictions:
            # 将分数转换为概率
            predictions = torch.cat(self.test_predictions).cpu()
            probabilities = torch.softmax(predictions, dim=1).numpy()
            
            # 保存预测结果到CSV
            submission_df = pd.DataFrame({
                'image_id': self.test_image_ids,
                'healthy': probabilities[:, 0],
                'multiple_diseases': probabilities[:, 1],
                'rust': probabilities[:, 2],
                'scab': probabilities[:, 3]
            })
            submission_df.to_csv('data/plant_pathodolgy_data/submission.csv', index=False)
            print("\nSubmission saved to: data/plant_pathodolgy_data/submission.csv")
            
            # 清理
            self.test_predictions.clear()
            self.test_image_ids.clear()

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                     step_size=self.hparams.lr_decay_steps,
                                     gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                 T_max=self.hparams.lr_decay_steps,
                                                 eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch"
                }
            }

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        #工厂方法创建损失函数
        self.loss_function = LossFactory(loss).build()

    def load_model(self):
        #动态获取模型模块
        name = self.hparams.model_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    # def instancialize(self, Model, **other_args):
    #     """ Instancialize a model using the corresponding parameters
    #         from self.hparams dictionary. You can also input any args
    #         to overwrite the corresponding value in self.hparams.
    #     """
    #     class_args = inspect.getfullargspec(Model.__init__).args[1:]
    #     inkeys = self.hparams.keys()
    #     args1 = {}
    #     for arg in class_args:
    #         if arg in inkeys:
    #             args1[arg] = getattr(self.hparams, arg)
    #     args1.update(other_args)
    #     return Model(**args1)
    def instancialize(self, Model, **other_args):
        """
        实例化模型：根据模型类 Model 的 __init__ 构造函数中的参数，
        从 self.hparams 中提取匹配的值，并调用 Model(...) 进行实例化。

        参数说明：
        - Model: 要实例化的模型类（如 GeneralBackboneClassifier）
        - other_args: 额外传入的参数，如果和 hparams 中有重名，会覆盖掉 hparams 的值

        返回：
        - Model 实例对象（即已经初始化好的模型）
        """

        # 1️⃣ 获取模型类构造函数的参数名（除了 self）
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        # 示例：['backbone_name', 'pretrained', 'num_classes', 'dropout_rate', ...]

        # 2️⃣ 获取 hparams 中有哪些 key（可理解为超参数的字段名）
        inkeys = self.hparams.keys()

        # 3️⃣ 遍历所有构造参数，如果在 hparams 中有对应的字段，就提取其值
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)  # hparams[arg]

        # 4️⃣ 如果调用 instancialize 时手动传入了一些额外参数，则用这些参数覆盖掉 hparams 中的对应项
        args1.update(other_args)

        # 5️⃣ 将整理好的参数传入模型类，完成实例化
        return Model(**args1)
