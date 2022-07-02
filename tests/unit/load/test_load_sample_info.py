from pathlib import Path

from sample_detection.load.scraper import load_sample_info


def test_load_sample_info():

    path = str(Path(__file__).resolve().parent / "test_files" / "sample_details.csv")
    sample_info = load_sample_info(path)

    assert all([isinstance(x, list) for x in sample_info["sample_in_times"]])
    assert all([isinstance(x, list) for x in sample_info["sample_from_times"]])
