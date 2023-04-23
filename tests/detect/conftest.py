import logging
import pytest
import numpy as np

from pathlib import Path
from tests.conftest import clip_length, sample_rate
from sample_detection.detect.embedding_generators.base import EmbeddingGenerator


class DummyEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, sample_duration, sample_rate):
        super().__init__(sample_duration=sample_duration, sample_rate=sample_rate)
        self.logger = logging.getLogger(__name__)

    def generate_embedding(self, audio: np.ndarray) -> np.ndarray:
        return np.array([np.float32(0) for i in range(512)])


@pytest.fixture()
def embedding_generator(clip_length, sample_rate):
    return DummyEmbeddingGenerator(sample_duration=clip_length, sample_rate=sample_rate)


@pytest.fixture()
def audio(sample_rate):
    return np.array([np.float32(0) for i in range(sample_rate * 60)])


@pytest.fixture()
def embedding():
    return [0 for i in range(512)]


@pytest.fixture()
def test_file_dir():
    return Path(__file__).resolve().parent.parent / "test_files"


@pytest.fixture()
def sample_details_dir(test_file_dir):
    return test_file_dir / "sample-details"


@pytest.fixture()
def sample_details_path(sample_details_dir):
    return sample_details_dir / "sample_details.csv"


@pytest.fixture()
def sample_details_without_one_to_one_match_path(sample_details_dir):
    return sample_details_dir / "sample_details_bad.csv"


@pytest.fixture()
def audio_dir(test_file_dir):
    return test_file_dir / "audio"


@pytest.fixture()
def fan_noise_path(audio_dir):
    return audio_dir / "fan_noise.mp3"


@pytest.fixture()
def ocean_noise_path(audio_dir):
    return audio_dir / "ocean_noise.mp3"


@pytest.fixture()
def rain_noise_path(audio_dir):
    return audio_dir / "rain_noise.mp3"


@pytest.fixture()
def white_noise_path(audio_dir):
    return audio_dir / "white_noise.mp3"
