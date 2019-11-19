import numpy as np
import pytest
import torch

from tests.base import (
    images_list, flows_list, NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH)
from flow_transforms import ToTensor, RandomAdditiveColor


@pytest.fixture
def get_tensor_data(images_list, flows_list):
    tt = ToTensor()
    images, flows = tt(images_list, flows_list)
    return images, flows


def test_add(get_tensor_data):
    images, flows = get_tensor_data
    ac = RandomAdditiveColor(stdev=1.0, independent=False)
    images2, flows2 = ac(images.clone(), flows)
    diff = images2[0, 0, 0, 0] - images[0, 0, 0, 0]
    assert torch.allclose(images+diff, images2)


def test_add_independent(get_tensor_data):
    images, flows = get_tensor_data
    ac = RandomAdditiveColor(stdev=1.0, independent=True)
    images2, flows2 = ac(images.clone(), flows)
    diff = images2[0, 0, 0, 0] - images[0, 0, 0, 0]
    assert not torch.allclose(images+diff, images2)
    for i in range(images.shape[0]):
        diff = images2[i, 0, 0, 0] - images[i, 0, 0, 0]
        assert torch.allclose(images[i]+diff, images2[i])
