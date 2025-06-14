from .factory import LossFactory
from .cross_entropy_onehot import CrossEntropyLossOneHot
# 如果将来还有其他，比如 FocalLoss, LabelSmoothingLoss，也可以导出来

__all__ = [
    'LossFactory',
    'CrossEntropyLossOneHot',
    # 'FocalLoss',
    # 'LabelSmoothingLoss',
]
