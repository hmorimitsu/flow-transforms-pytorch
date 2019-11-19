import numpy as np
import pytest
import torch

from tests.base import (
    images_list, flows_list, NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH)
from flow_transforms import ToTensor, RandomGammaColor


@pytest.fixture
def get_tensor_data(images_list, flows_list):
    tt = ToTensor()
    images, flows = tt(images_list, flows_list)
    return images, flows


def test_gamma(get_tensor_data):
    images, flows = get_tensor_data
    range_min = 0.75
    range_max = 1.5
    pixel_min = -10.0
    pixel_max = 10.0
    pixel_interval = pixel_max - pixel_min
    gc = RandomGammaColor(
        range_min=range_min, range_max=range_max,
        pixel_min=pixel_min, pixel_max=pixel_max, independent=False)
    for _ in range(10):
        images2, flows2 = gc(images.clone(), flows)
        exponent = (
            torch.log((images2[0, 0, 0, 0]-pixel_min)/pixel_interval) /
            torch.log((images[0, 0, 0, 0]-pixel_min)/pixel_interval))
        images_gamma = (
            torch.pow((images-pixel_min)/pixel_interval, exponent) *
            pixel_interval + pixel_min)
        assert torch.allclose(images_gamma, images2, atol=1e-5)
        assert exponent >= 1/range_max and exponent <= 1/range_min


def test_gamma_independent(get_tensor_data):
    images, flows = get_tensor_data
    range_min = 0.75
    range_max = 1.5
    pixel_min = -10.0
    pixel_max = 10.0
    pixel_interval = pixel_max - pixel_min
    gc = RandomGammaColor(
        range_min=range_min, range_max=range_max,
        pixel_min=pixel_min, pixel_max=pixel_max, independent=True)
    images2, flows2 = gc(images.clone(), flows)
    exponent = (
        torch.log((images2[0, 0, 0, 0]-pixel_min)/pixel_interval) /
        torch.log((images[0, 0, 0, 0]-pixel_min)/pixel_interval))
    images_gamma = (
        torch.pow((images-pixel_min)/pixel_interval, exponent) *
        pixel_interval + pixel_min)
    assert not torch.allclose(images_gamma, images2, atol=1e-5)
    for i in range(images.shape[0]):
        exponent = (
            torch.log((images2[i, 0, 0, 0]-pixel_min)/pixel_interval) /
            torch.log((images[i, 0, 0, 0]-pixel_min)/pixel_interval))
        images_gamma = (
            torch.pow((images[i]-pixel_min)/pixel_interval, exponent) *
            pixel_interval + pixel_min)
        assert torch.allclose(images_gamma, images2[i], atol=1e-5)
