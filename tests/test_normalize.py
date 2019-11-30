import numpy as np
import pytest
import torch

from tests.base import (
    images_list, flows_list, NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH)
from flow_transforms import ToTensor, Normalize


@pytest.fixture
def get_tensor_data(images_list, flows_list):
    tt = ToTensor()
    images, flows = tt(images_list, flows_list)
    return images, flows


def test_means(get_tensor_data):
    images, flows = get_tensor_data
    n = Normalize([0, 0, 0], [1, 1, 1])
    images2, flows2 = n(images.clone(), flows)
    assert torch.allclose(images, images2)
    assert torch.allclose(flows, flows2)

    means = [0.5, 1.0, 2.0]
    n = Normalize(means, [1, 1, 1])
    images2, flows2 = n(images.clone(), flows)
    for i in range(len(means)):
        assert torch.allclose(images[:, i]-means[i], images2[:, i])


def test_stdevs(get_tensor_data):
    images, flows = get_tensor_data
    n = Normalize([0, 0, 0], [0.5, 0.5, 0.5])
    images2, flows2 = n(images.clone(), flows)
    assert torch.allclose(images*2, images2)

    stdevs = [0.5, 1.0, 2.0]
    n = Normalize([0, 0, 0], stdevs)
    images2, flows2 = n(images.clone(), flows)
    for i in range(len(stdevs)):
        assert torch.allclose(images[:, i]/stdevs[i], images2[:, i])
