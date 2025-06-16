from model.losses.cross_entropy_onehot import CrossEntropyLossOneHot
from model.losses.focal_loss import FocalLoss
import torch.nn as nn

class LossFactory:
    def __init__(self, loss_type):
        self.loss_type = loss_type.lower()

    def build(self):
        if self.loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.loss_type == 'cross_entropy_onehot':
            return CrossEntropyLossOneHot()
        elif self.loss_type == 'soft_cross_entropy':
            return CrossEntropyLossOneHot()
        elif self.loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        elif self.loss_type == 'focal':
            return FocalLoss()
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')
