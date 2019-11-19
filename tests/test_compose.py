import numpy as np
import torch

from tests.base import (
    images_list, flows_list, NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH)
from flow_transforms import Compose, ToTensor


def test_compose(images_list, flows_list):
    c = Compose([ToTensor()])
    images, flows = c(images_list, flows_list)
    assert isinstance(images, torch.Tensor)
    assert isinstance(flows, torch.Tensor)
    assert images.shape == torch.Size(
        [NUM_IMAGES, 3, IMAGE_HEIGHT, IMAGE_WIDTH])
    assert flows.shape == torch.Size(
        [NUM_IMAGES-1, 2, IMAGE_HEIGHT, IMAGE_WIDTH])

    input_images = np.stack(images_list, axis=0)
    input_flows = np.stack(flows_list, axis=0)
    images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    flows = flows.detach().cpu().numpy().transpose(0, 2, 3, 1)
    assert np.allclose(images, input_images)
    assert np.allclose(flows, input_flows)
