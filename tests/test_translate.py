import numpy as np
import pytest
import torch

from tests.base import (
    images_list, flows_list, NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH)
from flow_transforms import ToTensor, RandomTranslate


@pytest.fixture
def get_tensor_data(images_list, flows_list):
    tt = ToTensor()
    images, flows = tt(images_list, flows_list)
    return images, flows


def test_translate(get_tensor_data):
    images, flows = get_tensor_data
    translation = (20, 20)
    rt = RandomTranslate(translation)
    images2, flows2 = rt(images, flows)
    assert (
        images.shape[2] >= images2.shape[2] and
        images.shape[3] >= images2.shape[3])
    assert (
        images2.shape[2] == flows2.shape[2] and
        images2.shape[3] == flows2.shape[3])
    th = images.shape[2] - images2.shape[2]
    tw = images.shape[3] - images2.shape[3]
    if torch.allclose(images[0, :, th:, :images.shape[3]-tw], images2[0]):
        tw = -tw
    elif torch.allclose(images[0, :, :images.shape[2]-th, tw:], images2[0]):
        th = -th
    elif torch.allclose(images[0, :, :images.shape[2]-th, :images.shape[3]-tw], images2[0]):
        th = -th
        tw = -tw

    flows[::2, 0] += tw
    flows[::2, 1] += th
    flows[1::2, 0] -= tw
    flows[1::2, 1] -= th

    x1, x2 = max(0, tw), min(images.shape[3]+tw, images.shape[3])
    y1, y2 = max(0, th), min(images.shape[2]+th, images.shape[2])
    assert torch.allclose(images[::2, :, y1:y2, x1:x2], images2[::2])
    assert torch.allclose(flows[::2, :, y1:y2, x1:x2], flows2[::2])

    x1, x2 = max(0, -tw), min(images.shape[3]-tw, images.shape[3])
    y1, y2 = max(0, -th), min(images.shape[2]-th, images.shape[2])
    assert torch.allclose(images[1::2, :, y1:y2, x1:x2], images2[1::2])
    assert torch.allclose(flows[1::2, :, y1:y2, x1:x2], flows2[1::2])
