import numpy as np
import torch

from tests.base import (
    images_list, flows_list, NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH)
from flow_transforms import ToTensor


def basic_checks(input_images,
                 input_flows,
                 images_order='HWC',
                 flows_order='HWC',
                 fp16=False,
                 device='cpu',
                 tolerance=1e-8):
    tt = ToTensor(
        images_order, flows_order, fp16, device)
    images, flows = tt(input_images, input_flows)
    assert isinstance(images, torch.Tensor)
    assert isinstance(flows, torch.Tensor)
    assert images.shape == torch.Size(
        [NUM_IMAGES, 3, IMAGE_HEIGHT, IMAGE_WIDTH])
    assert flows.shape == torch.Size(
        [NUM_IMAGES-1, 2, IMAGE_HEIGHT, IMAGE_WIDTH])

    input_images = np.stack(input_images, axis=0)
    input_flows = np.stack(input_flows, axis=0)
    if images_order == 'CHW':
        input_images = input_images.transpose(0, 2, 3, 1)
    if flows_order == 'CHW':
        input_flows = input_flows.transpose(0, 2, 3, 1)
    images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    flows = flows.detach().cpu().numpy().transpose(0, 2, 3, 1)
    assert np.allclose(images, input_images, atol=tolerance)
    assert np.allclose(flows, input_flows, atol=tolerance)


def test_cuda(images_list, flows_list):
    if torch.cuda.is_available:
        basic_checks(images_list, flows_list, device='cuda')
        basic_checks(
            images_list, flows_list, fp16=True, device='cuda', tolerance=1e-3)


def test_fp16(images_list, flows_list):
    basic_checks(images_list, flows_list, fp16=True, tolerance=1e-3)


def test_list(images_list, flows_list):
    basic_checks(images_list, flows_list)


def test_ndarray(images_list, flows_list):
    basic_checks(np.stack(images_list), np.stack(flows_list))


def test_tuple(images_list, flows_list):
    basic_checks(tuple(images_list), tuple(flows_list))


def test_images_chw(images_list, flows_list):
    images_list = np.stack(images_list, axis=0).transpose(0, 3, 1, 2)
    basic_checks(images_list, flows_list, images_order='CHW')


def test_flows_chw(images_list, flows_list):
    flows_list = np.stack(flows_list, axis=0).transpose(0, 3, 1, 2)
    basic_checks(images_list, flows_list, flows_order='CHW')


def test_both_chw(images_list, flows_list):
    images_list = np.stack(images_list, axis=0).transpose(0, 3, 1, 2)
    flows_list = np.stack(flows_list, axis=0).transpose(0, 3, 1, 2)
    basic_checks(
        images_list, flows_list, images_order='CHW', flows_order='CHW')
