import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.base import (
    images_list, flows_list, NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH)
from flow_transforms import ToTensor, RandomRotate


@pytest.fixture
def get_tensor_data(images_list, flows_list):
    tt = ToTensor()
    images, flows = tt(images_list, flows_list)
    return images, flows


def test_no_rotate(get_tensor_data):
    images, flows = get_tensor_data
    rr = RandomRotate(angle=0, diff_angle=0)
    images2, flows2 = rr(images, flows)
    assert torch.allclose(images, images2)
    assert torch.allclose(flows, flows2)


def test_rotate(get_tensor_data):
    images, flows = get_tensor_data
    rr = RandomRotate(angle=90, diff_angle=90)
    images2, flows2 = rr(images.clone(), flows.clone())
    # TODO: create better tests for rotation
    assert not torch.allclose(images, images2)
    assert not torch.allclose(flows, flows2)
    assert not torch.allclose(images2[0], images2[1])
    assert not torch.allclose(flows2[0], flows2[1])


def test_cuda(get_tensor_data):
    if torch.cuda.is_available:
        images, flows = get_tensor_data
        images = images.cuda()
        flows = flows.cuda()
        rr = RandomRotate(angle=0, diff_angle=0)
        images2, flows2 = rr(images.clone(), flows.clone())
        assert images2.is_cuda
        assert flows2.is_cuda


def test_fp16(get_tensor_data):
    if torch.cuda.is_available:
        images, flows = get_tensor_data
        images = images.half().cuda()
        flows = flows.half().cuda()
        rr = RandomRotate(angle=0, diff_angle=0)
        images2, flows2 = rr(images.clone(), flows.clone())
        assert images2.dtype == torch.float16
        assert flows2.dtype == torch.float16
