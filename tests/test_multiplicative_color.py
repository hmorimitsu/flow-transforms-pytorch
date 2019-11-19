import numpy as np
import pytest
import torch

from tests.base import (
    images_list, flows_list, NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH)
from flow_transforms import ToTensor, RandomMultiplicativeColor


@pytest.fixture
def get_tensor_data(images_list, flows_list):
    tt = ToTensor()
    images, flows = tt(images_list, flows_list)
    return images, flows


def test_mult(get_tensor_data):
    images, flows = get_tensor_data
    range_min = -2
    range_max = 5
    mc = RandomMultiplicativeColor(
        range_min=range_min, range_max=range_max, independent=False)
    for _ in range(10):
        images2, flows2 = mc(images.clone(), flows)
        scale = images2[0, 0, 0, 0] / images[0, 0, 0, 0]
        assert torch.allclose(images*scale, images2)
        assert scale >= range_min and scale <= range_max


def test_mult_independent(get_tensor_data):
    images, flows = get_tensor_data
    mc = RandomMultiplicativeColor(range_min=-2, range_max=5, independent=True)
    images2, flows2 = mc(images.clone(), flows)
    scale = images2[0, 0, 0, 0] / images[0, 0, 0, 0]
    assert not torch.allclose(images*scale, images2)
    for i in range(images.shape[0]):
        scale = images2[i, 0, 0, 0] / images[i, 0, 0, 0]
        assert torch.allclose(images[i]*scale, images2[i])
