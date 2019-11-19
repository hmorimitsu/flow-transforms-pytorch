import numpy as np
import pytest
import torch

from tests.base import (
    images_list, flows_list, NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH)
from flow_transforms import ToTensor, AddGaussianNoise


@pytest.fixture
def get_tensor_data(images_list, flows_list):
    tt = ToTensor()
    images, flows = tt(images_list, flows_list)
    return images, flows


def test_gauss(get_tensor_data):
    images, flows = get_tensor_data
    gn = AddGaussianNoise(stdev=1.0)
    images2, flows2 = gn(images.clone(), flows)
    assert not torch.allclose(images, images2)
    mean_diff = (images.mean() - images2.mean()).abs()
    assert mean_diff < 1
