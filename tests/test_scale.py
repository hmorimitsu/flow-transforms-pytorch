import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.base import (
    images_list, flows_list, NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH)
from flow_transforms import ToTensor, Scale


@pytest.fixture
def get_tensor_data(images_list, flows_list):
    tt = ToTensor()
    images, flows = tt(images_list, flows_list)
    return images, flows


def test_no_scale(get_tensor_data):
    images, flows = get_tensor_data
    s = Scale(images.shape[2:4])
    images2, flows2 = s(images, flows)
    assert torch.allclose(images, images2)
    assert torch.allclose(flows, flows2)


def test_scale_bigger(get_tensor_data):
    images, flows = get_tensor_data
    size = (images.shape[2]*2, images.shape[3])
    s = Scale(size)
    images2, flows2 = s(images, flows)
    images = F.interpolate(images, size, mode='bilinear', align_corners=False)
    flows = F.interpolate(flows, size, mode='bilinear', align_corners=False)
    flows[:, 0] *= 2
    assert torch.allclose(images, images2)
    assert torch.allclose(flows, flows2)

    images, flows = get_tensor_data
    size = (images.shape[2], images.shape[3]*2)
    s = Scale(size)
    images2, flows2 = s(images, flows)
    images = F.interpolate(images, size, mode='bilinear', align_corners=False)
    flows = F.interpolate(flows, size, mode='bilinear', align_corners=False)
    flows[:, 1] *= 2
    assert torch.allclose(images, images2)
    assert torch.allclose(flows, flows2)

    images, flows = get_tensor_data
    size = (int(images.shape[2]*1.5), images.shape[3]*2)
    s = Scale(size)
    images2, flows2 = s(images, flows)
    images = F.interpolate(images, size, mode='bilinear', align_corners=False)
    flows = F.interpolate(flows, size, mode='bilinear', align_corners=False)
    flows[:, 0] *= 1.5
    flows[:, 1] *= 2
    assert torch.allclose(images, images2)
    assert torch.allclose(flows, flows2)


def test_scale_smaller(get_tensor_data):
    images, flows = get_tensor_data
    size = (images.shape[2]//2, images.shape[3])
    s = Scale(size)
    images2, flows2 = s(images, flows)
    images = F.interpolate(images, size, mode='bilinear', align_corners=False)
    flows = F.interpolate(flows, size, mode='bilinear', align_corners=False)
    flows[:, 0] *= 0.5
    assert torch.allclose(images, images2)
    assert torch.allclose(flows, flows2)

    images, flows = get_tensor_data
    size = (images.shape[2], images.shape[3]//2)
    s = Scale(size)
    images2, flows2 = s(images, flows)
    images = F.interpolate(images, size, mode='bilinear', align_corners=False)
    flows = F.interpolate(flows, size, mode='bilinear', align_corners=False)
    flows[:, 1] *= 0.5
    assert torch.allclose(images, images2)
    assert torch.allclose(flows, flows2)

    images, flows = get_tensor_data
    size = (images.shape[2]//2, images.shape[3]//3)
    s = Scale(size)
    images2, flows2 = s(images, flows)
    images = F.interpolate(images, size, mode='bilinear', align_corners=False)
    flows = F.interpolate(flows, size, mode='bilinear', align_corners=False)
    flows[:, 0] *= 0.5
    flows[:, 1] *= 1.0/3
    assert torch.allclose(images, images2)
    assert torch.allclose(flows, flows2)
