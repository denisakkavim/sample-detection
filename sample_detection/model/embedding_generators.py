from abc import ABC, abstractmethod
import wav2clip
import numpy as np
import pandas as pd
import logging

from typing import Tuple, Dict

from sample_detection.model.sample_loader import SampleLoader


class EmbeddingGenerator(ABC):
    def __init__(self, sample_duration, sample_rate):
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True
        self.sample_loader = SampleLoader(
            sample_duration=sample_duration, sample_rate=sample_rate
        )

    @abstractmethod
    def generate_embedding(self, audio_array: np.ndarray) -> np.ndarray:

        """Generate an embedding from an audio clip (represented as a 1D ndarray).

        :param audio_array: 1D np.ndarray representation of an audio clip
        :type audio_array: np.ndarray

        :return: Embedding
        """
        return NotImplementedError()

    def generate_embeddings_from_directory(
        self, sample_info: pd.DataFrame, audio_dir: str
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[int, np.ndarray]]]:

        """Generate an embedding from an audio clip (represented as a 1D ndarray).

        :param sample_info: Sample info (as scraped from scraper), with sample times as lists
        :type sample_info: pd.DataFrame
        :param audio_dir: Directory containing the audio files scraped by the scraper.
        :type audio_dir: str

        :return: Dataframe of samples for which embeddings were succesfully generated, and a nested dictionary of
        embeddings, with filenames and sample start times as keys, and embeddings as values.
        """

        self.logger.info(f"Generating embeddings for audio in {audio_dir}")

        bad_samples = set()
        embeddings = {
            youtube_id: {}
            for youtube_id in (
                set(sample_info["sample_from_ytid"])
                | set(sample_info["sample_in_ytid"])
            )
        }

        for _, row in sample_info.iterrows():

            sample_good, sample_audio = self.sample_loader.load_sample(
                audio_dir=audio_dir, sample=row
            )

            if sample_good:

                sample_embeddings = {}

                for ytid in sample_audio:
                    sample_embeddings[ytid] = {
                        time: self.generate_embedding(audio)
                        for time, audio in sample_audio[ytid].items()
                    }

                for ytid in sample_embeddings:
                    embeddings[ytid] = {**embeddings[ytid], **sample_embeddings[ytid]}
            else:
                bad_samples.add(row["whosampled_id"])

        sample_info = sample_info[~sample_info["whosampled_id"].isin(bad_samples)]

        return sample_info, embeddings


class Wav2ClipEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, sample_duration, sample_rate):

        super().__init__(sample_duration=sample_duration, sample_rate=sample_rate)

        self.logger = logging.getLogger(__name__)
        self.model = wav2clip.get_model()

    def generate_embedding(self, audio_array: np.ndarray) -> np.ndarray:

        """Generate a wav2clip embedding from an audio clip (represented as a 1D ndarray).

        :return: Embedding
        """

        embedding = wav2clip.embed_audio(audio_array, self.model)
        embedding = np.squeeze(embedding)

        return embedding
