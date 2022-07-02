import numpy as np
import pytest


@pytest.fixture
def sample_rate():
    return 16000


@pytest.fixture
def audio(sample_rate):
    return np.array([0 for i in range(sample_rate * 60)])


@pytest.fixture
def start_time():
    return 0


@pytest.fixture
def clip_length():
    return 15
