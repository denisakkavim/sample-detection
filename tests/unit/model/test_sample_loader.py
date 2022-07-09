import pandas as pd
import pytest

from ast import literal_eval
from pathlib import Path

from sample_detection.model.sample_loader import SampleLoader

test_files_path = Path(__file__).resolve().parent / "test_files"


def test_load_audio(caplog, clip_length, sample_rate):

    sample_loader = SampleLoader(sample_duration=clip_length, sample_rate=sample_rate)

    audio_path = test_files_path / "audio" / "white_noise.mp3"
    loaded_audio = sample_loader._load_audio(audio_path=audio_path, start_time=0)

    assert (
        (loaded_audio.ndim == 1)
        and ("Loading audio from file" in caplog.text)
        and ("Successfully loaded audio from file" in caplog.text)
    )


def test_load_audio_file_does_not_exist(caplog, clip_length, sample_rate):

    sample_loader = SampleLoader(sample_duration=clip_length, sample_rate=sample_rate)

    audio_path = test_files_path / "audio" / "asdf.mp3"

    with pytest.raises(ValueError):
        loaded_audio = sample_loader._load_audio(audio_path=audio_path, start_time=0)

    assert ("Loading audio from file" in caplog.text) and (
        "Could not load audio from file" in caplog.text
    )


def test_load_sample(caplog, clip_length, sample_rate):

    sample_loader = SampleLoader(sample_duration=clip_length, sample_rate=sample_rate)

    test_files_path = Path(__file__).resolve().parent / "test_files"

    sample_info_path = test_files_path / "sample-details" / "sample_details.csv"
    audio_path = test_files_path / "audio"

    sample_info = pd.read_csv(sample_info_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    sample = sample_info.iloc[0, :]
    sample_loaded_successfully, loaded_sample = sample_loader.load_sample(
        audio_dir=audio_path, sample=sample
    )

    sample_in_ytid = sample["sample_in_ytid"]
    sample_from_ytid = sample["sample_from_ytid"]

    assert (
        (
            (sample_in_ytid in loaded_sample)
            and (sample_from_ytid in loaded_sample)
            and all(
                [
                    time in loaded_sample[sample_in_ytid]
                    for time in sample["sample_in_times"]
                ]
            )
            and all(
                [
                    time in loaded_sample[sample_from_ytid]
                    for time in sample["sample_from_times"]
                ]
            )
            and sample_loaded_successfully
        )
        and ("Verifying sample with whosampled_id" in caplog.text)
        and ("passed verification" in caplog.text)
    )


def test_load_sample_no_one_to_one_sample_match(caplog, clip_length, sample_rate):

    sample_loader = SampleLoader(sample_duration=clip_length, sample_rate=sample_rate)

    test_files_path = Path(__file__).resolve().parent / "test_files"

    sample_info_path = test_files_path / "sample-details" / "sample_details_bad.csv"
    audio_path = test_files_path / "audio"

    sample_info = pd.read_csv(sample_info_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    sample = sample_info.iloc[1, :]
    sample_loaded_successfully, loaded_sample = sample_loader.load_sample(
        audio_dir=audio_path, sample=sample
    )

    assert (sample_loaded_successfully == False) and (
        "Could not find a one-to-one match" in caplog.text
    )


def test_load_sample_cannot_load_sample(caplog, clip_length, sample_rate):

    sample_loader = SampleLoader(sample_duration=clip_length, sample_rate=sample_rate)

    test_files_path = Path(__file__).resolve().parent / "test_files"

    sample_info_path = test_files_path / "sample-details" / "sample_details_bad.csv"
    audio_path = test_files_path / "audio"

    sample_info = pd.read_csv(sample_info_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    sample = sample_info.iloc[0, :]
    sample_loaded_successfully, loaded_sample = sample_loader.load_sample(
        audio_dir=audio_path, sample=sample
    )

    assert (sample_loaded_successfully == False) and (
        "Could not load sample instance" in caplog.text
    )
