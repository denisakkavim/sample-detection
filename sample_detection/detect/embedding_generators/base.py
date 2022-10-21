import librosa
import logging
import numpy as np
import os
import pandas as pd
import warnings

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, Union

SampleAudio = Dict[int, np.ndarray]  # key is start time, value is np.array of audio
SampleInstance = Dict[str, Union[None, SampleAudio]]  # key is ytid of song

SampleAudioEmbedding = Dict[int, np.ndarray]
CollatedSampleAudioEmbeddings = Dict[str, SampleAudioEmbedding]


class EmbeddingGenerator(ABC):
    def __init__(
        self,
        sample_rate: int,
        sample_duration: int = 15,
    ):

        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True

        self.sample_rate = sample_rate
        self.sample_duration = sample_duration

    def _load_audio(self, audio_path: str, start_time: int) -> np.ndarray:

        """Load an audioclip.

        :param audio_path: Path to file containing the audio for the sample to be loaded.
        :type audio_path: str
        :param start_time: Start time (in seconds) of the sample
        :type start_time: int

        :raises e: ValueError, FileNotFoundError
        :return: np.ndarray representation of the audio in audioclip

        """

        if isinstance(audio_path, Path):
            audio_path = str(audio_path)

        self.logger.info(
            f"Loading audio from file at {audio_path}, between {start_time}s and {start_time + self.sample_duration} seconds"
        )
        try:

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio_array, _ = librosa.load(
                    path=audio_path,
                    sr=self.sample_rate,
                    offset=start_time,
                    duration=self.sample_duration,
                )
            self.logger.info(
                f"Successfully loaded audio from file at {audio_path}, between {start_time}s and {start_time + self.sample_duration} seconds"
            )
            if len(audio_array) != self.sample_rate * self.sample_duration:
                raise ValueError(
                    "Audio array is shorter than it should be - will cause issues with embedding generators."
                )
            return audio_array
        except (ValueError, FileNotFoundError) as e:
            self.logger.info(
                f"Could not load audio from file at {audio_path}, between {start_time}s and {start_time + self.sample_duration} seconds"
            )
            raise e

    def load_sample(
        self, audio_dir: str, sample: pd.Series
    ) -> Tuple[bool, SampleInstance]:

        """Load audio for both songs in a sample.

        :param sample: Row from sample info dataframe (as scraped by scraper), with sample times as lists
        :type sample: pd.Series
        :param audio_dir: Directory containing the audio files scraped by the scraper.
        :type audio_dir: str

        :return: Whether the sample was loaded successfully, and a nested dictionary of embeddings, with filenames and sample start times as keys, and audio arrays as values.
        """

        self.logger.info(f'Loading sample with whosampled_id {sample["whosampled_id"]}')

        if len(sample["sample_in_times"]) != len(sample["sample_from_times"]):
            self.logger.warning(
                f'Sample with whosampled id {sample["whosampled_id"]} could not be loaded and will not be used. Reason: Could not find a one-to-one match between sample instances. '
            )
            return False, None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio_dict = {
                    sample["sample_from_ytid"]: {
                        time: self._load_audio(
                            audio_path=os.path.join(
                                audio_dir, f'{sample["sample_from_ytid"]}.mp3'
                            ),
                            start_time=time,
                        )
                        for time in sample["sample_from_times"]
                    },
                    sample["sample_in_ytid"]: {
                        time: self._load_audio(
                            audio_path=os.path.join(
                                audio_dir, f'{sample["sample_in_ytid"]}.mp3'
                            ),
                            start_time=time,
                        )
                        for time in sample["sample_in_times"]
                    },
                }

            self.logger.info(
                f'Sample with whosampled_id {sample["whosampled_id"]} successfully loaded '
            )
            return True, audio_dict

        except (ValueError, FileNotFoundError):
            self.logger.warning(
                f'Sample with whosampled_id {sample["whosampled_id"]} could not be loaded and will not be used. Reason: Could not load sample instance. '
            )
            return False, None

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
    ) -> Tuple[pd.DataFrame, CollatedSampleAudioEmbeddings]:

        """Generate an embedding from an audio clip (represented as a 1D ndarray).

        :param sample_info: Sample info (as scraped from scraper), with sample times as lists
        :type sample_info: pd.DataFrame
        :param audio_dir: Directory containing the audio files scraped by the scraper.
        :type audio_dir: str

        :return: Dataframe of samples for which embeddings were succesfully generated, and a nested dictionary of embeddings, with filenames and sample start times as keys, and embeddings as values.
        """

        self.logger.info(f"Generating embeddings for audio in {audio_dir}")

        bad_samples = set()
        embeddings = dict()

        for _, row in sample_info.iterrows():

            sample_good, sample_audio = self.load_sample(
                audio_dir=audio_dir, sample=row
            )

            if sample_good:

                sample_embeddings = {}

                for ytid in sample_audio:
                    sample_embeddings[ytid] = {
                        time: self.generate_embedding(audio_array)
                        for time, audio_array in sample_audio[ytid].items()
                    }

                for ytid in sample_embeddings:
                    embeddings[ytid] = {
                        **embeddings.get(ytid, dict()),
                        **sample_embeddings.get(ytid, dict()),
                    }
            else:
                bad_samples.add(row["whosampled_id"])

        # check that all ytids have associated embeddings, and remove any that don't:
        for key, value in embeddings.items():
            if not value:
                embeddings.pop(key)

        sample_info = sample_info[~sample_info["whosampled_id"].isin(bad_samples)]

        return sample_info, embeddings
