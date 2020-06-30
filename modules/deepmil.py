
import torch as torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim=2048, num_classes=2):
        super(Attention, self).__init__()
        self.L = hidden_dim
        # self.D = int(hidden_dim / 2)
        self.D = 128
        self.K = 1
        self.classes = num_classes
        self.A_grad = None

        # --- transformation f #

        # -- What exactly is this attention? Is this gated attention? Normal attention?
        # -- This is normal attention. Gated attention adds an element-wise multiplication with a linear layer and a sigmoid non-linearity
        self.attention = nn.Sequential( # in : batch_size * L
            nn.Linear(self.L, self.D),  # per tile, gives 1*D
            nn.Tanh(),
            nn.Linear(self.D, self.K)   # per tile, gives 1*K
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.classes),
            # nn.Sigmoid() # Since we use the torch crossentropy class we need no sigmoid here
        )

    def forward(self, H):
        """
        Takes an bag of extracted feature vectors and classifies them accordingly
        """
        H = H.squeeze(0) # Since we get a 1 x #instances x #dimensions, as batch_size = 1

        # x = x.squeeze(0)
        # H = self.feature_extractor_part1(x)
        # H = H.view(-1, 48 * 6 * 6)
        # H = self.feature_extractor_part2(H)  # --> NxL = batch_size * 512
        A = self.attention(H)  # NxK         # = batch_size * 1 -- thus
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        if self.train and H.requires_grad:
            A.register_hook(self._set_grad(A)) # save grad specifically here. only when in train mode and when grad's computed

        M = torch.mm(A, H)  # KxL

        Y_out = self.classifier(M)
        Y_hat = Y_out.softmax(dim=1).argmax() # take softmax over each batch. In this case, we have batch = 1
        return Y_out, Y_hat, A

    # internal function to extract grad
    def _set_grad(self, var):
        def hook(grad):
            self.A_grad = grad
        return hook

    # AUXILIARY METHODS
    def calculate_classification_error(self, Y, Y_hat):
        Y = Y.float()
        # _, Y_hat, _ = self.forward(X)
        # print(Y_hat)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat
