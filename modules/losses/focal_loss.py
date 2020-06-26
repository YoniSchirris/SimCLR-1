
import torch

class FocalLoss():
    def __init__(self, alpha, gamma):
        self.a = alpha
        self.g = gamma
    
    def __call__(self, loss):
        # Expects a CE loss of size=batch_size
        pt = torch.exp(-loss)
        return (self.a * ((1.0-pt) ** self.g) * loss).mean()
