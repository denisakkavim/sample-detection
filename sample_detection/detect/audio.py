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
            if audio.ndim != 1:
                raise ValueError(
                    "1D array expected as input - flatten audio array to one dimension and try again"
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

    def get_numpy(self) -> np.ndarray:

        """Get the numpy representation of the audio clip.

        :return: Numpy representation of the audio clip
        :rtype: np.ndarray

        """
        return self.np_array

    def get_extract(self, start_time: int, extract_length: int) -> "Audio":

        """Get an extract of the current audio clip.

        :param start_time: Start time (in seconds) of the extract you want to get
        :type start_time: int
        :param extract_length: Length (in seconds) of the extract you want to get
        :type extract_length: int

        :return: Extract of the current audio clip

        """

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
                "Extract length is longer than the remaining audio. Padding remaining audio with zeros."
            )
            audio_to_end = self.np_array[self.sample_rate * start_time :]
            padding = np.zeros(
                shape=(self.sample_rate * extract_length - len(audio_to_end))
            )
            return Audio(audio=np.concatenate([audio_to_end, padding]))
