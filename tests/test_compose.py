import numpy as np
import torch

from tests.base import (
    images_list, flows_list, NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH)
import flow_transforms


def base_compose_tranform(fp16, cuda):
    transform = flow_transforms.Compose([
        flow_transforms.ToTensor(fp16=fp16, cuda=cuda),
        flow_transforms.RandomTranslate((10, 10)),
        flow_transforms.RandomRotate(10, 10),
        flow_transforms.RandomScale(0.9, 1.1),
        flow_transforms.RandomCrop((50, 80)),
        flow_transforms.CenterCrop((40, 70)),
        flow_transforms.RandomGammaColor(0.7, 1.5, 0, 1),
        flow_transforms.AddGaussianNoise(10),
        flow_transforms.RandomAdditiveColor(1),
        flow_transforms.RandomMultiplicativeColor(0.5, 2.0),
        flow_transforms.RandomVerticalFlip(),
        flow_transforms.RandomHorizontalFlip()
    ])
    return transform


def test_basic_compose(images_list, flows_list):
    c = flow_transforms.Compose([flow_transforms.ToTensor()])
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


def test_cuda(images_list, flows_list):
    if torch.cuda.is_available:
        c = base_compose_tranform(False, True)
        images, flows = c(images_list, flows_list)
        assert images.is_cuda
        assert flows.is_cuda


def test_cuda(images_list, flows_list):
    if torch.cuda.is_available:
        c = base_compose_tranform(True, True)
        images, flows = c(images_list, flows_list)
        assert images.is_cuda
        assert flows.is_cuda
        assert images.dtype == torch.float16
        assert flows.dtype == torch.float16
