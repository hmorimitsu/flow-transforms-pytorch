import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.base import (
    images_list, flows_list, NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH)
from flow_transforms import ToTensor, RandomScale


@pytest.fixture
def get_tensor_data(images_list, flows_list):
    tt = ToTensor()
    images, flows = tt(images_list, flows_list)
    return images, flows


def test_scale_bigger(get_tensor_data):
    images, flows = get_tensor_data
    range_min = 1.1
    range_max = 1.5
    mc = RandomScale(range_min=range_min, range_max=range_max)
    images2, flows2 = mc(images, flows)
    assert (
        images.shape[2] < images2.shape[2] and
        images.shape[3] < images2.shape[3])
    assert (
        images2.shape[2] == flows2.shape[2] and
        images2.shape[3] == flows2.shape[3])
    flows_resized = F.interpolate(
        flows, size=flows2.shape[2:4], mode='bilinear', align_corners=False)
    size_scale = float(images2.shape[2]) / images.shape[2]
    flow_scale = flows2[0, 0, 0, 0] / flows_resized[0, 0, 0, 0]
    assert abs(size_scale - flow_scale) < 1e-2


def test_scale_smaller(get_tensor_data):
    images, flows = get_tensor_data
    range_min = 0.5
    range_max = 0.9
    mc = RandomScale(range_min=range_min, range_max=range_max)
    images2, flows2 = mc(images, flows)
    assert (
        images.shape[2] > images2.shape[2] and
        images.shape[3] > images2.shape[3])
    assert (
        images2.shape[2] == flows2.shape[2] and
        images2.shape[3] == flows2.shape[3])
    flows_resized = F.interpolate(
        flows, size=flows2.shape[2:4], mode='bilinear', align_corners=False)
    size_scale = float(images2.shape[2]) / images.shape[2]
    flow_scale = flows2[0, 0, 0, 0] / flows_resized[0, 0, 0, 0]
    assert abs(size_scale - flow_scale) < 1e-2
