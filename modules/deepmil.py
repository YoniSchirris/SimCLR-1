
import torch as torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim=2048, num_classes=2):
        super(Attention, self).__init__()
        self.L = hidden_dim
        self.D = int(hidden_dim / 2)
        self.K = 1
        self.classes = num_classes
        self.A_grad = None

        # --- transformation f #

        self.feature_extractor_part1 = nn.Sequential(  # # in = 32x32x3

            # In classiﬁcation, morphology of nuclei (shape, size, color, and texture) is necessary
            # to distinguish between different types of them. Raw RGB color intensities which constitute the overall
            # visual appearance of nuclei were, thus, chosen as input features to softmax CNN for each patch
            nn.Conv2d(3, 36, kernel_size=5),  # Change to RGB input --> # 28x28x36
            # filter dimensions: 5x5x3x36
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # --> # 14x14x20
            # filter dimension: 2x2

            nn.Conv2d(36, 48, kernel_size=3),  # --> # 12x12x48
            # filter dimensions: 3x3x36x8
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # --> 6x6x48

        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(48 * 6 * 6, self.L),  # change to 6x6 as input image is slightl larger --> 1x512
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(self.L, self.L),  # -> 1x512
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # --- end of transformation f

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

        A.register_hook(self._set_grad(A)) # save grad specifically here

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

    def calculate_objective(self, X, Y, Y_prob, A):
        Y = Y.float()

        # X = X.view(X.shape[0], X.shape[3], X.shape[1], X.shape[2]) # 1 x num_tiles x rgb x width x height
        # Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5).squeeze(0)    # Y_prob is 1x2, we want it to be of size 2 for later dot product

        # Y_prob is now 2-dimensional
        # Y is actually 1-dimensional

        Y_onehot = torch.zeros(2)

        Y_onehot[int(Y.item())] = 1 # Y is either 0 or 1, but is wrapped inside a tensor and is a float

        Y_onehot.to(device=Y_prob.device.type)  # set Y_onehot to same device as Y_prob

        neg_log_likelihood = -1 * Y_onehot.dot(torch.log(Y_prob)) # 1x1

        return neg_log_likelihood, A