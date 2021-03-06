import torchvision
from modules.transformations.colour_normalization import MyHETransform

class TransformsBYOL:
    def __init__(self, size, henorm='', path_to_target_im='', lut_root_dir=''):
        """
        Appendix B, page 13, bottom:
        In both training and evaluation, we normalize color channels by subtracting the average color and dividing by the standar deviation, 
        computed on ImageNet, after applying the augmentations
        """
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )

        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((196.99545288085938, 174.21389770507812, 208.27586364746094), (47.20835494995117, 63.753047943115234, 44.500614166259766))
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor()
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)




class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly 
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size, henorm='', path_to_target_im='', lut_root_dir=''):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                MyHETransform(henorm=henorm, path_to_target_im=path_to_target_im, lut_root_dir=lut_root_dir),
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                MyHETransform(henorm=henorm, path_to_target_im=path_to_target_im, lut_root_dir=lut_root_dir),
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor()
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)
