# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import torch
import numpy as np
import elasticdeform.torch as etorch


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ElasticDeform(object):
    """Elastic Deformation helper class"""

    def __init__(self, control_points_num=3, sigma=15, axis=(1, 2)):
        self.control_points_num = control_points_num
        self.sigma = sigma
        self.axis = axis

    def __call__(self, x):
        # generate a deformation grid
        displacement = np.random.randn(2, self.control_points_num, self.control_points_num) * self.sigma
        # construct PyTorch input and top gradient
        displacement = torch.tensor(displacement)
        # elastic deformation
        ed_x = etorch.deform_grid(x, displacement, prefilter=True, axis=self.axis)
        return ed_x
