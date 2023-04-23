import logging
import numpy as np

from pathlib import Path

from sample_detection.detect.sample_detector import SampleDetector
from sample_detection.detect.audio import Audio

test_files_dir = Path(__file__).resolve().parent.parent / "test_files"
audio_dir = test_files_dir / "audio"


def test_init(caplog, clip_length):
    with caplog.at_level(logging.INFO):
        model = SampleDetector(sample_duration=clip_length)
    assert "Setting random seed" in caplog.text


def test_fit(sample_info, clip_length, min_negatives):
    model = SampleDetector(sample_duration=clip_length)
    model.fit(sample_info=sample_info, audio_dir=audio_dir, min_negatives=min_negatives)

    try:
        model.predict(np.array([0 for i in range(1024)]))
    except:
        pass

    assert True


def test_predict_both_audio(
    sample_info, clip_length, min_negatives, fan_noise_path, rain_noise_path
):
    model = SampleDetector(sample_duration=clip_length)

    model.fit(sample_info=sample_info, audio_dir=audio_dir, min_negatives=min_negatives)

    audio_1 = Audio(path=fan_noise_path, clip_length=clip_length)
    audio_2 = Audio(path=rain_noise_path, clip_length=clip_length)

    pred = model.predict(audio_1=audio_1, audio_2=audio_2)

    assert (pred >= 0) and (pred <= 1)


def test_predict_both_embeddings(sample_info, clip_length, min_negatives, embedding):
    model = SampleDetector(sample_duration=clip_length)
    model.fit(sample_info=sample_info, audio_dir=audio_dir, min_negatives=min_negatives)

    pred = model.predict(embedding_1=embedding, embedding_2=embedding)

    assert (pred >= 0) and (pred <= 1)


def test_predict_mixed_audio_and_embedding(
    sample_info, clip_length, min_negatives, embedding, fan_noise_path
):
    model = SampleDetector(sample_duration=clip_length)
    model.fit(sample_info=sample_info, audio_dir=audio_dir, min_negatives=min_negatives)

    audio = Audio(path=fan_noise_path, clip_length=clip_length)
    pred = model.predict(audio_1=audio, embedding_2=embedding)

    assert (pred >= 0) and (pred <= 1)


def test_predict_both_embeddings_logging(
    caplog, sample_info, clip_length, min_negatives, fan_noise_path, embedding
):
    model = SampleDetector(sample_duration=clip_length)
    model.fit(sample_info=sample_info, audio_dir=audio_dir, min_negatives=min_negatives)

    audio = Audio(path=fan_noise_path, clip_length=clip_length)

    with caplog.at_level(logging.INFO):
        pred = model.predict(
            audio_1=audio, embedding_1=embedding, embedding_2=embedding
        )

    assert "Both audio and an embedding have been passed in" in caplog.text


def test_sample_detection(
    sample_info, clip_length, min_negatives, fan_noise_path, rain_noise_path
):
    model = SampleDetector(sample_duration=clip_length)
    model.fit(sample_info=sample_info, audio_dir=audio_dir, min_negatives=min_negatives)

    audio_1 = Audio(path=fan_noise_path, clip_length=clip_length)
    audio_2 = Audio(path=rain_noise_path, clip_length=clip_length)

    found_samples = model.find_samples(audio_1=audio_1, audio_2=audio_2, threshold=0)

    assert all(
        [
            (key in sample)
            for key in ["start_time_1", "start_time_2", "confidence"]
            for sample in found_samples
        ]
    )
