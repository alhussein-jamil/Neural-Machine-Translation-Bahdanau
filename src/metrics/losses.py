import torch
import torch.nn.functional as F
from torch import nn


class Loss(nn.Module):
    def __init__(self, loss_fn) -> None:
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if isinstance(self.loss_fn, nn.CrossEntropyLoss):
            y = y.long()
            y_idx = F.one_hot(y, num_classes=x.shape[-1]).float()
            losses = self.loss_fn(x, y_idx)
        elif isinstance(self.loss_fn, nn.NLLLoss):
            _, _, vocab_size = x.shape
            x = x.view(-1, vocab_size)
            #apply softmax to x 
            x = F.log_softmax(x, dim=1)
            y = y.view(-1).long()
            losses = self.loss_fn(x, y)
        else:
            raise ValueError("Unsupported loss function type")
        return losses.mean()
