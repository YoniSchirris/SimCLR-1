
import torch as torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim=2048, intermediate_hidden_dim=128, num_classes=2, attention_bias=True):
        super(Attention, self).__init__()
        self.L = hidden_dim
        # self.D = int(hidden_dim / 2)
        self.D = intermediate_hidden_dim
        self.K = 1
        self.classes = num_classes
        self.A_grad = None

        # --- transformation f #

        # -- What exactly is this attention? Is this gated attention? Normal attention?
        # -- This is normal attention. Gated attention adds an element-wise multiplication with a linear layer and a sigmoid non-linearity
        self.attention = nn.Sequential( # in : batch_size * L
            nn.Linear(self.L, self.D, bias=attention_bias),  # per tile, gives 1*D
            nn.Tanh(),
            nn.Linear(self.D, self.K, bias=attention_bias)   # per tile, gives 1*K
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.classes),
            # nn.Sigmoid() # Since we use the torch crossentropy class we need no sigmoid here
        )

    def forward(self, H):
        """
        Takes an bag of extracted feature vectors and classifies them accordingl
        
        """
        # print(f"Shape of H being passed: {H.shape}")
        #TODO CHANGE BACK, THIS WAS FOR THE KATHER DATA MSI TEST
        H = H.permute(0,2,1) # (batch x channels x instances) -> (batch x instances x channels)
        # print(f"Shape of H being passed after permutation: {H.shape}")

        # We pass a (batch x channels) x instances tensor into attention network, which does a tile-wise attention computation. This is then reshaped back to represen the batches
        A = self.attention(H.reshape(-1, H.shape[-1])).reshape((H.shape[0], H.shape[1], 1))  # A = (batch x instances x K=1)
        A = F.softmax(A, dim=1)  # softmax over # instances. A = (batch x instances x K=1)

        if self.train and H.requires_grad:
            A.register_hook(self._set_grad(A)) # save grad specifically here. only when in train mode and when grad's computed

        M = torch.einsum('abc, abd -> ad', A, H) # (batch x instances x K=1) * (batch x instances x channels) -> (batch x instances). We end up with a weighted average feature vectors per patient

        Y_out = self.classifier(M) # (batch x channels) -> (batch x num_classes)

        if self.classes > 1:
            # When doing logistic regression
            Y_hat = Y_out.argmax(dim=1) # (batch x num_classes) -> (batch x 1)
        else:
            # When doing linear regression
            Y_hat = Y_out

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



class AttentionWithStd(nn.Module):
    def __init__(self, hidden_dim=2048, intermediate_hidden_dim=128, num_classes=2, attention_bias=True):
        super(AttentionWithStd, self).__init__()
        self.L = hidden_dim
        # self.D = int(hidden_dim / 2)
        self.D = intermediate_hidden_dim
        self.K = 1
        self.classes = num_classes
        self.A_grad = None

        # --- transformation f #

        # -- What exactly is this attention? Is this gated attention? Normal attention?
        # -- This is normal attention. Gated attention adds an element-wise multiplication with a linear layer and a sigmoid non-linearity
        self.attention = nn.Sequential( # in : batch_size * L
            nn.Linear(self.L, self.D, bias=attention_bias),  # per tile, gives 1*D
            nn.Tanh(),
            nn.Linear(self.D, self.K, bias=attention_bias)   # per tile, gives 1*K
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * self.L * self.K, self.classes),
            # nn.Sigmoid() # Since we use the torch crossentropy class we need no sigmoid here
        )

    def compute_weighted_std(self, A, H, M):
        # Following https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
        # A: Attention (weight):    batch x instances x 1
        # H: Hidden:                batch x instances x channels
        H = H.permute(0,2,1)      # batch x channels x instances
           
        # M: Weighted average:      batch x channels
        M = M.unsqueeze(dim=2)    # batch x channels x 1
        # ---> S: weighted stdev:   batch x channels

        # N is non-zero weights for each bag: batch x 1

        N = (A != 0).sum(dim=1)

        upper = torch.einsum('abc, adb -> ad', A, (H - M)**2) # batch x channels
        lower = ((N-1) * torch.sum(A, dim=1)) / N # batch x 1

        S = torch.sqrt(upper / lower)

        return S
 
    def forward(self, H):
        """
        Takes an bag of extracted feature vectors and classifies them accordingl
        
        """
        H = H.permute(0,2,1) # (batch x channels x instances) -> (batch x instances x channels)

        # We pass a (batch x channels) x instances tensor into attention network, which does a tile-wise attention computation. This is then reshaped back to represen the batches
        A = self.attention(H.reshape(-1, H.shape[-1])).reshape((H.shape[0], H.shape[1], 1))  # A = (batch x instances x K=1)
        A = F.softmax(A, dim=1)  # softmax over # instances. A = (batch x instances x K=1)

        if self.train and H.requires_grad:
            A.register_hook(self._set_grad(A)) # save grad specifically here. only when in train mode and when grad's computed

        M = torch.einsum('abc, abd -> ad', A, H) # (batch x instances) * (batch x instances x channels) -> (batch x channels). We end up with a weighted average feature vectors per patient

        S = self.compute_weighted_std(A, H, M)

        MS = torch.cat((M,S), dim=1) # concatenate the two tensors among the feature dimension, giving a twice as big feature

        Y_out = self.classifier(MS) # (batch x channels) -> (batch x num_classes)

        if self.classes > 1:
            # When doing logistic regression
            Y_hat = Y_out.argmax(dim=1) # (batch x num_classes) -> (batch x 1)
        else:
            # When doing linear regression
            Y_hat = Y_out

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
