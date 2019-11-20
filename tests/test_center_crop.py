import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.base import (
    images_list, flows_list, NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH)
from flow_transforms import ToTensor, CenterCrop


@pytest.fixture
def get_tensor_data(images_list, flows_list):
    tt = ToTensor()
    images, flows = tt(images_list, flows_list)
    return images, flows


def test_crop_size(get_tensor_data):
    images, flows = get_tensor_data
    margins = (20, 30)
    size = (images.shape[2] - margins[0], images.shape[3] - margins[1])
    cc = CenterCrop(size)
    images2, flows2 = cc(images, flows)
    assert (
        images.shape[2] == images2.shape[2]+margins[0] and
        images.shape[3] == images2.shape[3]+margins[1])
    assert (
        images2.shape[2] == flows2.shape[2] and
        images2.shape[3] == flows2.shape[3])
    assert torch.allclose(
        images2, images[:, :, margins[0]//2:-margins[0]//2, margins[1]//2:-margins[1]//2])
    assert torch.allclose(
        flows2, flows[:, :, margins[0]//2:-margins[0]//2, margins[1]//2:-margins[1]//2])


def test_no_crop(get_tensor_data):
    images, flows = get_tensor_data
    margins = (0, 0)
    size = (images.shape[2] - margins[0], images.shape[3] - margins[1])
    cc = CenterCrop(size)
    images2, flows2 = cc(images, flows)
    assert torch.allclose(images2, images)
    assert torch.allclose(flows2, flows)
