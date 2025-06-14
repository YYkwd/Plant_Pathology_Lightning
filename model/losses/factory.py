import torch.nn.functional as F
from model.losses.cross_entropy_onehot import CrossEntropyLossOneHot  # 你的自定义 loss

class LossFactory:
    def __init__(self, loss_name):
        self.loss_name = loss_name.lower()

    def build(self):
        if self.loss_name == 'mse':
            return F.mse_loss
        elif self.loss_name == 'l1':
            return F.l1_loss
        elif self.loss_name == 'bce':
            return F.binary_cross_entropy
        elif self.loss_name in ('soft_ce', 'soft_cross_entropy'):
            return CrossEntropyLossOneHot()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_name}")
