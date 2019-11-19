import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.base import (
    images_list, flows_list, NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH)
from flow_transforms import ToTensor, RandomCrop


@pytest.fixture
def get_tensor_data(images_list, flows_list):
    tt = ToTensor()
    images, flows = tt(images_list, flows_list)
    return images, flows


def test_crop_size(get_tensor_data):
    images, flows = get_tensor_data
    margins = (20, 30)
    size = (images.shape[2] - margins[0], images.shape[3] - margins[1])
    rc = RandomCrop(size)
    images2, flows2 = rc(images, flows)
    assert (
        images.shape[2] == images2.shape[2]+margins[0] and
        images.shape[3] == images2.shape[3]+margins[1])
    assert (
        images2.shape[2] == flows2.shape[2] and
        images2.shape[3] == flows2.shape[3])


def test_crop_vals(get_tensor_data):
    images, flows = get_tensor_data
    margins = (1, 1)
    size = (images.shape[2] - margins[0], images.shape[3] - margins[1])
    rc = RandomCrop(size)
    images2, flows2 = rc(images, flows)
    if torch.allclose(images2, images[:, :, :-1, :-1]):
        assert torch.allclose(flows2, flows[:, :, :-1, :-1])
    elif torch.allclose(images2, images[:, :, 1:, :-1]):
        assert torch.allclose(flows2, flows[:, :, 1:, :-1])
    elif torch.allclose(images2, images[:, :, :-1, 1:]):
        assert torch.allclose(flows2, flows[:, :, :-1, 1:])
    elif torch.allclose(images2, images[:, :, 1:, 1:]):
        assert torch.allclose(flows2, flows[:, :, 1:, 1:])


def test_no_crop(get_tensor_data):
    images, flows = get_tensor_data
    margins = (0, 0)
    size = (images.shape[2] - margins[0], images.shape[3] - margins[1])
    rc = RandomCrop(size)
    images2, flows2 = rc(images, flows)
    assert torch.allclose(images2, images)
    assert torch.allclose(flows2, flows)
