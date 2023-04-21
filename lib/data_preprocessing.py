from PIL import ImageFilter
import random

class MultiTransforms:
    """Take multiple crops of one image."""

    def __init__(self, base_transforms):
        if not isinstance(base_transforms, (list, tuple)):
            base_transforms = [base_transforms]
        self.base_transforms = base_transforms

    def __call__(self, x):
        crops = []
        for t in self.base_transforms:
            crops.append(t(x))
        return crops

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

if __name__ == '__main__':
    pass