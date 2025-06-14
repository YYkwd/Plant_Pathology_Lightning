# # @Author: yican, yelanlan
# # @Date: 2020-06-16 20:43:36
# # @Last Modified by:   yican
# # @Last Modified time: 2020-06-14 16:21:14
# # Third party libraries
# import torch.nn as nn
# import torch


# class CrossEntropyLossOneHot(nn.Module):
#     def __init__(self):
#         super(CrossEntropyLossOneHot, self).__init__()
#         self.log_softmax = nn.LogSoftmax(dim=-1)

#     def forward(self, preds, labels):
#         return torch.mean(torch.sum(-labels * self.log_softmax(preds), -1))
# @Author: ChatGPT Refactor Team
# @Date: 2025-04-19
# Refactored for PyTorch 2.4+

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLossOneHot(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        """
        preds: [batch_size, num_classes] (raw logits)
        labels: [batch_size, num_classes] (float or one-hot labels)

        Automatically handles:
        - soft labels (float)
        - hard labels (one-hot)
        """
        # preds shape: [batch_size, num_classes]
        # labels shape: [batch_size, num_classes]

        log_probs = F.log_softmax(preds, dim=-1)
        loss = -(labels * log_probs).sum(dim=-1).mean()
        return loss
