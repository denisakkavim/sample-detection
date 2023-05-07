from sample_detection.scrape.load import load_sample_info


def test_load_sample_info(sample_details_path):
    sample_info = load_sample_info(sample_details_path)

    assert all(
        [isinstance(x, list) for x in sample_info["sample_in_times"]]
        + [isinstance(x, list) for x in sample_info["sample_from_times"]]
    )
