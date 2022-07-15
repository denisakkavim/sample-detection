import numpy as np
import pytest


@pytest.fixture
def audio(sample_rate):
    return np.array([np.float32(0) for i in range(sample_rate * 60)])
