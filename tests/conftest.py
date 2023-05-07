from ast import literal_eval
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_rate():
    return 16000


@pytest.fixture
def start_time():
    return 0


@pytest.fixture
def clip_length():
    return 15


@pytest.fixture
def test_file_dir():
    return Path(__file__).resolve().parent / "test_files"


@pytest.fixture
def sample_details_dir(test_file_dir):
    return test_file_dir / "sample-details"


@pytest.fixture()
def sample_details_path(sample_details_dir):
    return sample_details_dir / "sample_details.csv"


@pytest.fixture
def min_negatives():
    return 1


@pytest.fixture
def sample_info(sample_details_path):
    sample_info = pd.read_csv(sample_details_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    return sample_info
