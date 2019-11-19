import numpy as np
import pytest

NUM_IMAGES = 3
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 150


@pytest.fixture
def images_list():
    images = [np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
              for _ in range(NUM_IMAGES)]
    return images


@pytest.fixture
def flows_list():
    flows = [np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH, 2)
             for _ in range(NUM_IMAGES-1)]
    return flows
