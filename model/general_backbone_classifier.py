import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# L2归一化函数
def l2_norm(x, dim=1, eps=1e-12):
    return F.normalize(x, p=2, dim=dim, eps=eps)

# 分类头模块
class BinaryHead(nn.Module):
    def __init__(self, num_classes=4, emb_size=2048, scale=1.0):
        super().__init__()
        self.scale = scale
        self.fc = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = l2_norm(x)            # 归一化特征
        logits = self.fc(x) * self.scale  # 全连接后乘缩放系数
        return logits

# 通用分类模型（支持自定义backbone、dropout开关、返回特征向量）
class GeneralBackboneClassifier(nn.Module):
    def __init__(self,
                 backbone_name='seresnext50_32x4d',
                 pretrained=False,
                 num_classes=4,
                 dropout_rate=0.0,
                 return_embedding=False,
                 scale=1.0):
        """
        - backbone_name: timm支持的backbone名字
        - pretrained: 是否加载预训练权重
        - num_classes: 分类类别数
        - dropout_rate: dropout比例 (0.0表示不启用)
        - return_embedding: 是否同时输出embedding特征
        - scale: 分类头缩放因子
        """
        super().__init__()

        # 加载骨干网络
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True
        )

        # 获取backbone最后一层输出通道数
        self.out_channels = self.backbone.feature_info.channels()[-1]

        # 自适应平均池化，将特征图压成(1,1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 扁平化后进行BatchNorm
        self.bn = nn.BatchNorm1d(self.out_channels)
        self.bn.bias.requires_grad_(False)

        # Dropout
        self.use_dropout = dropout_rate > 0.0
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

        # 分类头
        self.binary_head = BinaryHead(num_classes=num_classes,
                                      emb_size=self.out_channels,
                                      scale=scale)

        # 是否返回embedding
        self.return_embedding = return_embedding

    def forward(self, x):
        # 1. 特征提取
        features = self.backbone(x)[-1]

        # 2. 空间特征压缩
        x = self.avg_pool(features)
        x = x.view(x.size(0), -1)

        # 3. 特征归一化
        x = self.bn(x)

        # 4. (可选) dropout
        if self.use_dropout:
            x = self.dropout(x)

        # 5. 分类
        logits = self.binary_head(x)

        if self.return_embedding:
            # 同时返回logits和embedding特征
            return logits, x
        else:
            # 只返回logits
            return logits
