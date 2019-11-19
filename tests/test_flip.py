import numpy as np
import pytest
import torch

from tests.base import (
    images_list, flows_list, NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH)
from flow_transforms import (
    ToTensor, RandomHorizontalFlip, RandomVerticalFlip)


@pytest.fixture
def get_tensor_data(images_list, flows_list):
    tt = ToTensor()
    images, flows = tt(images_list, flows_list)
    return images, flows


def test_horizontal(get_tensor_data):
    images, flows = get_tensor_data
    num_runs = 10
    hf = RandomHorizontalFlip()
    for _ in range(num_runs):
        images2, flows2 = hf(images, flows)
        if torch.allclose(images, images2):
            assert torch.allclose(flows, flows2)
        else:
            assert torch.allclose(images, torch.flip(images2, dims=[3]))
            flows2[:, 0] *= -1
            assert torch.allclose(flows, torch.flip(flows2, dims=[3]))


def test_vertical(get_tensor_data):
    images, flows = get_tensor_data
    num_runs = 10
    vf = RandomVerticalFlip()
    for _ in range(num_runs):
        images2, flows2 = vf(images, flows)
        if torch.allclose(images, images2):
            assert torch.allclose(flows, flows2)
        else:
            assert torch.allclose(images, torch.flip(images2, dims=[2]))
            flows2[:, 1] *= -1
            assert torch.allclose(flows, torch.flip(flows2, dims=[2]))
