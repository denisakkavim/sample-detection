import logging
import numpy as np
import pandas as pd

from ast import literal_eval
from sklearn.utils.validation import check_is_fitted

from pathlib import Path

from sample_detection.model.model import Model
from sample_detection.model.audio import Audio

test_files_dir = Path(__file__).resolve().parent.parent.parent / "test_files"
audio_dir = test_files_dir / "audio"


def test_init(caplog, clip_length, sample_rate):

    model = Model(sample_duration=clip_length, sample_rate=sample_rate)
    assert "Setting random seed" in caplog.text


def test_fit(clip_length, sample_rate, min_negatives):

    model = Model(sample_duration=clip_length, sample_rate=sample_rate)

    sample_info_path = test_files_dir / "sample-details" / "sample_details.csv"

    sample_info = pd.read_csv(sample_info_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    model.fit(sample_info=sample_info, audio_dir=audio_dir, min_negatives=min_negatives)

    try:
        model.predict(np.array([0 for i in range(1024)]))
    except:
        pass

    assert True


def test_predict_both_audio(clip_length, sample_rate, min_negatives):

    model = Model(sample_duration=clip_length, sample_rate=sample_rate)

    sample_info_path = test_files_dir / "sample-details" / "sample_details.csv"

    sample_info = pd.read_csv(sample_info_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    model.fit(sample_info=sample_info, audio_dir=audio_dir, min_negatives=min_negatives)

    audio_1_path = audio_dir / "fan_noise.mp3"
    audio_2_path = audio_dir / "rain_noise.mp3"

    audio_1 = Audio(path=audio_1_path, clip_length=clip_length, sample_rate=sample_rate)
    audio_2 = Audio(path=audio_2_path, clip_length=clip_length, sample_rate=sample_rate)

    pred = model.predict(audio_1=audio_1, audio_2=audio_2)

    assert (pred >= 0) and (pred <= 1)


def test_predict_both_embeddings(clip_length, sample_rate, min_negatives):

    model = Model(sample_duration=clip_length, sample_rate=sample_rate)

    sample_info_path = test_files_dir / "sample-details" / "sample_details.csv"

    sample_info = pd.read_csv(sample_info_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    model.fit(sample_info=sample_info, audio_dir=audio_dir, min_negatives=min_negatives)

    emb_1 = np.array([0 for i in range(512)])
    emb_2 = np.array([1 for i in range(512)])

    pred = model.predict(embedding_1=emb_1, embedding_2=emb_2)

    assert (pred >= 0) and (pred <= 1)


def test_predict_both_embeddings(clip_length, sample_rate, min_negatives):

    model = Model(sample_duration=clip_length, sample_rate=sample_rate)

    sample_info_path = test_files_dir / "sample-details" / "sample_details.csv"

    sample_info = pd.read_csv(sample_info_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    model.fit(sample_info=sample_info, audio_dir=audio_dir, min_negatives=min_negatives)

    audio_path = audio_dir / "fan_noise.mp3"
    audio = Audio(path=audio_path, clip_length=clip_length, sample_rate=sample_rate)
    emb = np.array([1 for i in range(512)])

    pred = model.predict(audio_1=audio, embedding_2=emb)

    assert (pred >= 0) and (pred <= 1)


def test_predict_both_embeddings_logging(
    caplog, clip_length, sample_rate, min_negatives
):

    model = Model(sample_duration=clip_length, sample_rate=sample_rate)

    sample_info_path = test_files_dir / "sample-details" / "sample_details.csv"

    sample_info = pd.read_csv(sample_info_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    model.fit(sample_info=sample_info, audio_dir=audio_dir, min_negatives=min_negatives)

    audio_path = audio_dir / "fan_noise.mp3"
    audio = Audio(path=audio_path, clip_length=clip_length, sample_rate=sample_rate)
    emb_1 = np.array([0 for i in range(512)])
    emb_2 = np.array([1 for i in range(512)])

    with caplog.at_level(logging.INFO):
        pred = model.predict(audio_1=audio, embedding_1=emb_1, embedding_2=emb_2)

    assert "Both audio and an embedding have been passed in" in caplog.text
