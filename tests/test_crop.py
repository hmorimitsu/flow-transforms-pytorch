import numpy as np
import pytest
import torch
import torch.nn.functional as F
import warnings

from tests.base import (
    images_list, flows_list, NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH)
from flow_transforms import ToTensor, Crop


@pytest.fixture
def get_tensor_data(images_list, flows_list):
    tt = ToTensor()
    images, flows = tt(images_list, flows_list)
    return images, flows


def test_errors(get_tensor_data):
    images, flows = get_tensor_data
    with pytest.raises(AssertionError):
        c = Crop((-1, 0), (1, 1), False)
    with pytest.raises(AssertionError):
        c = Crop((0, -1), (1, 1), False)
    with pytest.raises(AssertionError):
        c = Crop((0, 0), (0, 1), False)
    with pytest.raises(AssertionError):
        c = Crop((0, 0), (1, 0), False)
    with pytest.raises(AssertionError):
        c = Crop((-1, -1), (0, 0), False)


def test_warnings(get_tensor_data):
    images, flows = get_tensor_data
    c = Crop((0, 1), (2, 2), True)
    with pytest.warns(RuntimeWarning):
        images2, flows2 = c(images, flows)
    c = Crop((1, 0), (2, 2), True)
    with pytest.warns(RuntimeWarning):
        images2, flows2 = c(images, flows)
    c = Crop((images.shape[2], images.shape[3]-1), (2, 2), True)
    with pytest.warns(RuntimeWarning):
        images2, flows2 = c(images, flows)
    c = Crop((images.shape[2]-1, images.shape[3]), (2, 2), True)
    with pytest.warns(RuntimeWarning):
        images2, flows2 = c(images, flows)


def test_no_crop(get_tensor_data):
    images, flows = get_tensor_data
    c = Crop((0, 0), (images.shape[2], images.shape[3]), False)
    images2, flows2 = c(images, flows)
    assert torch.allclose(images, images2)
    assert torch.allclose(flows, flows2)


def test_crop(get_tensor_data):
    images, flows = get_tensor_data
    c = Crop((1, 1), (images.shape[2]-2, images.shape[3]-2), False)
    images2, flows2 = c(images, flows)
    assert torch.allclose(images[:, :, 1:-1, 1:-1], images2)
    assert torch.allclose(flows[:, :, 1:-1, 1:-1], flows2)


def test_oversize_crop(get_tensor_data):
    images, flows = get_tensor_data
    c = Crop((images.shape[2]//2, images.shape[3]//2), (2*images.shape[2], 2*images.shape[3]), True)
    with pytest.warns(RuntimeWarning):
        images2, flows2 = c(images, flows)
    assert torch.allclose(images, images2)
    assert torch.allclose(flows, flows2)
