import torch as torch
import torch.nn as nn
import torch.nn.functional as F

# https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/11
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

# https://medium.com/@mswang12/depthwise-separable-convolutions-simple-image-classification-with-pytorch-7f7d2ba06af7
model_d = nn.Sequential(OrderedDict([
    ('conv1_depthwise', nn.Conv2d(1, 1, 3, stride=2, padding=1, groups=1)),
    ('conv1_pointwise', nn.Conv2d(1, hidden_units[0], 1)),
    ('Relu1', nn.ReLU()),
    ('conv2_depthwise', nn.Conv2d(hidden_units[0], hidden_units[0], 3, stride=2, padding=1, groups=hidden_units[0])),
    ('conv2_pointwise', nn.Conv2d(hidden_units[0], hidden_units[1], 1)),
    ('Relu2', nn.ReLU()),
    ('conv3_depthwise', nn.Conv2d(hidden_units[1], hidden_units[1], 3, stride=2, padding=1, groups=hidden_units[1])),
    ('conv3_pointwise', nn.Conv2d(hidden_units[1], hidden_units[2], 1)),
    ('Relu3', nn.ReLU()),
    ('conv4_depthwise', nn.Conv2d(hidden_units[2], hidden_units[2], 4, stride=4, padding=0, groups=hidden_units[2])),
    ('conv4_pointwise', nn.Conv2d(hidden_units[2], output_units, 1)),
    ('log_softmax', nn.LogSoftmax(dim = 1))
]))



class CustomCNN(nn.Module):
    def __init__(self, seperable=False):
        """A custom CNN architecture to do compresesd WSI classification

        Args:
            seperable (bool, optional): [description]. Defaults to False.
        """
        super(CustomCNN, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = 8
        self.filter_depth = 128

        self.classifier = nn.Sequential(
           #...The complete CNN architecture consisted of 8 convolutional layers using strided 
           # depthwise separable convolutions with 128 3 × 3 filters, batch normalization (BN), 
           # leaky-ReLU acti- vation (LRA), L2 regularization with 1 × 10−5 coefficient, feature-wise 
           # 20% dropout, and stride of 2 except for the 7-th and 8-th layers with no stride; followed 
           # by a dense layer with 128 units, BN and LRA; and a final layer that depended on the application: 
           # a softmax dense layer for classification problems, and a linear output unit for regression tasks.

           #...We trained the CNN using stochastic gradient descent with Adam optimization and 16- sample mini-batch, 
           # decreasing the learning rate by a factor of 10 starting from 1 × 10−2 every time the validation metric 
           # plateaued until 1 × 10-5. We minimized MSE for regression, CE for classification,  
           
        )

    def forward(self, H):
        """
        Takes compressed images H and output a prediction
        """

        stuff = self.classifier(H)
        
        return stuff
