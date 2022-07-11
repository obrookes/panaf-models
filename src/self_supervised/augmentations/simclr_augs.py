import torch
import math
import numpy as np
from torch import nn
from kornia.augmentation import (
    ColorJitter,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomGrayscale,
    Resize,
)
from kornia.filters import gaussian_blur2d
from einops import rearrange


class SimCLRTrainDataTransform(nn.Module):
    def __init__(
        self,
        input_height: int = 224,
        gaussian_blur: bool = True,
        jitter_strength: float = 1.0,
    ) -> None:

        super().__init__()

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur

        self.transforms = nn.Sequential(
            RandomResizedCrop(
                size=(self.input_height, self.input_height), same_on_batch=True
            ),
            RandomHorizontalFlip(p=0.5, same_on_batch=True),
            ColorJitter(
                0.8 * self.jitter_strength,
                0.8 * self.jitter_strength,
                0.8 * self.jitter_strength,
                0.2 * self.jitter_strength,
                p=0.8,
                same_on_batch=True,
            ),
            RandomGrayscale(p=0.2, same_on_batch=True),
            GaussianBlur(kernel_size=0.1 * self.input_height),
        )

    def forward(self, sample):
        with torch.no_grad():
            b_size, t, c, w, h = sample.shape
            x = rearrange(sample, "b t c w h -> (b t) c w h")

            x_i = self.transforms(x)
            x_j = self.transforms(x)

            x_i = rearrange(x_i, "(b t) c w h -> b c t w h", b=b_size)
            x_j = rearrange(x_j, "(b t) c w h -> b c t w h", b=b_size)

        return x_i, x_j


class GaussianBlur(nn.Module):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):

        super().__init__()

        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = gaussian_blur2d(
                sample,
                (math.ceil(self.kernel_size), math.ceil(self.kernel_size)),
                (sigma, sigma),
            )

        return sample


class SimCLREvalDataTransform(nn.Module):
    def __init__(
        self,
        input_height: int = 224,
    ):
        super().__init__()
        self.input_height = input_height
        self.transforms = nn.Sequential(Resize((self.input_height, self.input_height)))

    def forward(self, sample):
        with torch.no_grad():
            b_size, t, c, w, h = sample.shape
            x = rearrange(sample, "b t c w h -> (b t) c w h")

            x_i = self.transforms(x)
            x_j = self.transforms(x)

            x_i = rearrange(x_i, "(b t) c w h -> b c t w h", b=b_size)
            x_j = rearrange(x_j, "(b t) c w h -> b c t w h", b=b_size)

        return x_i, x_j
