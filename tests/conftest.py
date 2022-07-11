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
def min_negatives():
    return 1


@pytest.fixture
def sample_info():

    sample_info_path = (
        Path(__file__).resolve().parent
        / "test_files"
        / "sample-details"
        / "sample_details.csv"
    )

    sample_info = pd.read_csv(sample_info_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    return sample_info
