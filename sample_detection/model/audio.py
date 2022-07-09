import librosa
import logging
import numpy as np

from pathlib import Path
from typing import Optional


class Audio:
    def __init__(
        self,
        path: Optional[str] = None,
        audio: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = 16000,
        start_time: Optional[str] = 0,
        clip_length: Optional[int] = None,
    ):

        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True

        self.sample_rate = sample_rate

        if isinstance(path, Path):
            path = str(path)

        if audio is not None:
            if path is not None:
                self.logger.warning(
                    "Both audio (ndarray) and path to file are passed as arguments: will use ndarray representation of audio instead of file at path"
                )
            self.np_array = audio
        elif path is not None:
            self.np_array, _ = librosa.load(
                path=path,
                sr=sample_rate,
                offset=start_time,
                duration=clip_length,
            )
        else:
            raise ValueError("One of path or audio must be specified")

        self.length = int(len(self.np_array) / self.sample_rate)

    def __len__(self):
        return self.length

    def get_numpy(self):
        return self.np_array

    def get_extract(self, start_time, extract_length):
        if self.sample_rate * (start_time + extract_length) < len(self.np_array):
            return Audio(
                audio=self.np_array[
                    self.sample_rate
                    * start_time : self.sample_rate
                    * (start_time + extract_length)
                ]
            )
        else:
            self.logger.warning(
                "Extract length is longer than the remaining audio. Extract will go to the end of the audio file."
            )
            return Audio(audio=self.np_array[self.sample_rate * start_time :])
