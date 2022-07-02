import librosa
import logging
import numpy as np
import os
import pandas as pd
import warnings

from typing import Dict, Tuple, Union


class SampleLoader:
    def __init__(self, sample_duration: int, sample_rate: int):
        self.logger = logging.getLogger(__name__)
        self.sample_duration = sample_duration
        self.sample_rate = sample_rate

    def _load_audio(self, audio_path: str, start_time: int) -> np.ndarray:
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
        except ValueError as e:
            self.logger.info(
                f"Could not load audio from file at {audio_path}, between {start_time}s and {start_time + self.sample_duration} seconds"
            )
            raise e

    def load_sample(
        self, audio_dir: str, sample: pd.Series
    ) -> Tuple[bool, Dict[str, Union[None, Dict[int, np.ndarray]]]]:

        self.logger.info(
            f'Verifying sample with whosampled_id {sample["whosampled_id"]}'
        )

        if len(sample["sample_in_times"]) != len(sample["sample_from_times"]):
            self.logger.warning(
                f'Sample with whosampled id {sample["whosampled_id"]} failed verification and will not be used. Reason: Could not find a one-to-one match between sample instances. '
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
                f'Sample with whosampled_id {sample["whosampled_id"]} passed verification'
            )
            return True, audio_dict

        except ValueError:
            self.logger.warning(
                f'Sample with whosampled_id {sample["whosampled_id"]} failed verification and will not be used. Reason: Could not load sample instance.'
            )
            return False, None
