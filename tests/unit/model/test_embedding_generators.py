import logging
import numpy as np
import pandas as pd

from ast import literal_eval
from pathlib import Path

from sample_detection.model.embedding_generators import (
    EmbeddingGenerator,
    Wav2ClipEmbeddingGenerator,
)


class DummyEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, sample_duration, sample_rate):

        super().__init__(sample_duration=sample_duration, sample_rate=sample_rate)
        self.logger = logging.getLogger(__name__)

    def generate_embedding(self, audio: np.ndarray) -> np.ndarray:
        return np.array([np.float32(0) for i in range(512)])


def test_generate_embeddings_from_directory(caplog, clip_length, sample_rate):

    embedding_generator = DummyEmbeddingGenerator(
        sample_duration=clip_length, sample_rate=sample_rate
    )

    test_files_path = Path(__file__).resolve().parent / "test_files"

    sample_info_path = test_files_path / "sample-details" / "sample_details.csv"
    audio_path = test_files_path / "audio"

    sample_info = pd.read_csv(sample_info_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    sample_info, embeddings = embedding_generator.generate_embeddings_from_directory(
        sample_info=sample_info, audio_dir=str(audio_path)
    )

    sample_file_names = set(sample_info["sample_from_ytid"]) | set(
        sample_info["sample_in_ytid"]
    )

    df = pd.concat(
        [
            sample_info[["sample_in_ytid", "sample_in_times"]].rename(
                mapper={"sample_in_ytid": "ytid", "sample_in_times": "times"}, axis=1
            ),
            sample_info[["sample_from_ytid", "sample_from_times"]].rename(
                mapper={"sample_from_ytid": "ytid", "sample_from_times": "times"},
                axis=1,
            ),
        ]
    )

    assert (
        ("Generating embeddings" in caplog.text)
        and all([file_name in sample_file_names for file_name in embeddings.keys()])
        and all(
            [(time in embeddings[row["ytid"]])]
            for _, row in df.iterrows()
            for time in row["times"]
        )
    )


def test_wav2clip_embedding_generator_init(clip_length, sample_rate):

    embedding_generator = Wav2ClipEmbeddingGenerator(
        sample_duration=clip_length, sample_rate=sample_rate
    )

    assert True


def test_wav2clip_generate_embedding(clip_length, sample_rate, audio):

    embedding_generator = Wav2ClipEmbeddingGenerator(
        sample_duration=clip_length, sample_rate=sample_rate
    )

    embedding = embedding_generator.generate_embedding(audio=audio)

    assert embedding.shape == (512,)
