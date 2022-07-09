import numpy as np
from pathlib import Path

from sample_detection.model.audio import Audio


def test_init_from_numpy(audio):

    audioclip = Audio(audio=audio)
    assert True


def test_length(audio):

    audioclip = Audio(audio=audio)
    assert len(audioclip) == 60


def test_init_from_path():

    path = str(
        Path(__file__).resolve().parent / "test_files" / "audio" / "white_noise.mp3"
    )
    audioclip = Audio(path=path)
    assert True


def test_init_both_path_and_audio(caplog, audio):

    path = str(
        Path(__file__).resolve().parent / "test_files" / "audio" / "white_noise.mp3"
    )
    audioclip = Audio(path=path, audio=audio)

    assert np.array_equal(audio, audioclip.get_numpy()) and (
        "Both audio (ndarray) and path to file are passed as arguments" in caplog.text
    )


def test_init_subclip(start_time, clip_length):

    path = str(
        Path(__file__).resolve().parent / "test_files" / "audio" / "white_noise.mp3"
    )
    audioclip = Audio(path=path, start_time=start_time, clip_length=clip_length)
    assert len(audioclip) == clip_length


def test_get_numpy(audio):

    audioclip = Audio(audio=audio)
    assert np.array_equal(audio, audioclip.get_numpy())


def test_get_extract(start_time, clip_length):

    path = str(
        Path(__file__).resolve().parent / "test_files" / "audio" / "white_noise.mp3"
    )

    audioclip = Audio(path=path)
    extract = audioclip.get_extract(start_time=start_time, extract_length=clip_length)
    assert len(extract) == clip_length


def test_get_extract_after_eof(caplog, clip_length):

    path = str(
        Path(__file__).resolve().parent / "test_files" / "audio" / "white_noise.mp3"
    )

    audioclip = Audio(path=path)
    extract = audioclip.get_extract(
        start_time=len(audioclip) - int(clip_length / 2), extract_length=clip_length
    )
    assert (len(extract) < clip_length) and (
        "Extract length is longer than the remaining audio." in caplog.text
    )
