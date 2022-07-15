import logging
import pandas as pd

from ast import literal_eval
from pathlib import Path

from sample_detection.detect.sample_loader import SampleLoader

test_files_dir = Path(__file__).resolve().parent.parent.parent / "test_files"
audio_dir = test_files_dir / "audio"


def test_load_audio(clip_length, sample_rate):

    sample_loader = SampleLoader(sample_duration=clip_length, sample_rate=sample_rate)

    audio_path = test_files_dir / "audio" / "white_noise.mp3"
    loaded_audio = sample_loader._load_audio(audio_path=audio_path, start_time=0)

    assert loaded_audio.ndim == 1


def test_load_audio_logging(caplog, clip_length, sample_rate):

    sample_loader = SampleLoader(sample_duration=clip_length, sample_rate=sample_rate)
    audio_path = test_files_dir / "audio" / "white_noise.mp3"

    with caplog.at_level(logging.INFO):
        loaded_audio = sample_loader._load_audio(audio_path=audio_path, start_time=0)

    assert ("Loading audio from file" in caplog.text) and (
        "Successfully loaded audio from file" in caplog.text
    )


def test_load_sample(clip_length, sample_rate):

    sample_loader = SampleLoader(sample_duration=clip_length, sample_rate=sample_rate)

    sample_info_path = test_files_dir / "sample-details" / "sample_details.csv"

    sample_info = pd.read_csv(sample_info_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    sample = sample_info.iloc[0, :]
    sample_loaded_successfully, loaded_sample = sample_loader.load_sample(
        audio_dir=audio_dir, sample=sample
    )

    sample_in_ytid = sample["sample_in_ytid"]
    sample_from_ytid = sample["sample_from_ytid"]

    assert (
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


def test_load_sample_logging(caplog, clip_length, sample_rate):

    sample_loader = SampleLoader(sample_duration=clip_length, sample_rate=sample_rate)

    sample_info_path = test_files_dir / "sample-details" / "sample_details.csv"

    sample_info = pd.read_csv(sample_info_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    sample = sample_info.iloc[0, :]
    with caplog.at_level(logging.INFO):
        sample_loaded_successfully, loaded_sample = sample_loader.load_sample(
            audio_dir=audio_dir, sample=sample
        )

    assert ("Loading sample with whosampled_id" in caplog.text) and (
        "successfully loaded" in caplog.text
    )


def test_load_sample_no_one_to_one_sample_match(clip_length, sample_rate):

    sample_loader = SampleLoader(sample_duration=clip_length, sample_rate=sample_rate)

    sample_info_path = test_files_dir / "sample-details" / "sample_details_bad.csv"

    sample_info = pd.read_csv(sample_info_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    sample = sample_info.iloc[1, :]
    sample_loaded_successfully, loaded_sample = sample_loader.load_sample(
        audio_dir=audio_dir, sample=sample
    )

    assert sample_loaded_successfully == False


def test_load_sample_no_one_to_one_sample_match_logging(
    caplog, clip_length, sample_rate
):

    sample_loader = SampleLoader(sample_duration=clip_length, sample_rate=sample_rate)

    sample_info_path = test_files_dir / "sample-details" / "sample_details_bad.csv"

    sample_info = pd.read_csv(sample_info_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    sample = sample_info.iloc[1, :]
    sample_loaded_successfully, loaded_sample = sample_loader.load_sample(
        audio_dir=audio_dir, sample=sample
    )

    assert "Could not find a one-to-one match" in caplog.text


def test_load_sample_cannot_load_sample(clip_length, sample_rate):

    sample_loader = SampleLoader(sample_duration=clip_length, sample_rate=sample_rate)

    sample_info_path = test_files_dir / "sample-details" / "sample_details_bad.csv"

    sample_info = pd.read_csv(sample_info_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    sample = sample_info.iloc[0, :]
    sample_loaded_successfully, loaded_sample = sample_loader.load_sample(
        audio_dir=audio_dir, sample=sample
    )

    assert sample_loaded_successfully == False


def test_load_sample_cannot_load_sample_logging(caplog, clip_length, sample_rate):

    sample_loader = SampleLoader(sample_duration=clip_length, sample_rate=sample_rate)

    sample_info_path = test_files_dir / "sample-details" / "sample_details_bad.csv"

    sample_info = pd.read_csv(sample_info_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    sample = sample_info.iloc[0, :]

    try:
        with caplog.at_level(logging.WARNING):
            sample_loaded_successfully, loaded_sample = sample_loader.load_sample(
                audio_dir=audio_dir, sample=sample
            )
    except:
        pass

    assert "Could not load sample instance" in caplog.text
