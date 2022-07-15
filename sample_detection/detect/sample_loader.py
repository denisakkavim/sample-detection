import librosa
import logging
import numpy as np
import os
import pandas as pd
import warnings

from pathlib import Path
from typing import Dict, Tuple, Union


class SampleLoader:
    def __init__(self, sample_duration: int, sample_rate: int):
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True
        self.sample_duration = sample_duration
        self.sample_rate = sample_rate

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
            audio, _ = librosa.load(
                path=audio_path,
                sr=self.sample_rate,
                offset=start_time,
                duration=self.sample_duration,
            )
            self.logger.info(
                f"Successfully loaded audio from file at {audio_path}, between {start_time}s and {start_time + self.sample_duration} seconds"
            )
            return audio
        except (ValueError, FileNotFoundError) as e:
            self.logger.info(
                f"Could not load audio from file at {audio_path}, between {start_time}s and {start_time + self.sample_duration} seconds"
            )
            raise e

    def load_sample(
        self, audio_dir: str, sample: pd.Series
    ) -> Tuple[bool, Dict[str, Union[None, Dict[int, np.ndarray]]]]:

        """Load audio for both songs in a sample.

        :param sample: Row from sample info dataframe (as scraped by scraper), with sample times as lists
        :type sample: pd.Series
        :param audio_dir: Directory containing the audio files scraped by the scraper.
        :type audio_dir: str

        :return: Whether the sample was loaded successfully, and a nested dictionary of
        embeddings, with filenames and sample start times as keys, and audio arrays as values.
        """

        self.logger.info(f'Loading sample with whosampled_id {sample["whosampled_id"]}')

        if len(sample["sample_in_times"]) != len(sample["sample_from_times"]):
            self.logger.warning(
                f'Sample with whosampled id {sample["whosampled_id"]} could not be loaded and will not be used. Reason: Could not find a one-to-one match between sample instances. '
            )
            return False, None

        try:
            with warnings.catch_warnings():
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
