import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        计算多标签Focal Loss，支持soft labels
        
        Args:
            inputs: 模型输出的logits，形状为 [batch_size, num_classes]
            targets: 目标标签，可以是hard labels (0/1) 或 soft labels (0~1之间的概率)
                    形状为 [batch_size, num_classes]
        """
        # 将logits转换为概率（用于调试和可视化）
        probs = torch.sigmoid(inputs)
        
        # 计算每个样本每个类别的损失
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算调制因子
        pt = torch.exp(-bce_loss)
        
        # 应用Focal Loss公式
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        # 对每个样本的所有类别取平均
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss 