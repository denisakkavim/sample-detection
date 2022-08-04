import logging
import numpy as np
import warnings

from sample_detection.detect.embedding_generators.base import EmbeddingGenerator
from sample_detection.detect.embedding_generators.musicnn._model import extractor


class MusicNNMaxPoolEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, sample_duration, sample_rate=16000):

        super().__init__(sample_duration=sample_duration, sample_rate=sample_rate)

        self.logger = logging.getLogger(__name__)

    def generate_embedding(self, audio_array: np.ndarray) -> np.ndarray:

        """Generate a musicnn max-pool embedding from an audio clip (represented as a 1D ndarray).

        :return: Embedding
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            _, _, features = extractor(
                audio=audio_array,
                sr=self.sample_rate,
                model="MTT_musicnn",
                input_length=self.sample_duration,
                input_overlap=False,
                extract_features=True,
            )

        embedding = np.array(features["max_pool"]).flatten()

        return embedding


class MusicNNMeanPoolEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, sample_duration, sample_rate=16000):

        super().__init__(sample_duration=sample_duration, sample_rate=sample_rate)

        self.logger = logging.getLogger(__name__)

    def generate_embedding(self, audio_array: np.ndarray) -> np.ndarray:

        """Generate a musicnn mean-pool embedding from an audio clip (represented as a 1D ndarray).

        :return: Embedding
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            _, _, features = extractor(
                audio=audio_array,
                sr=self.sample_rate,
                model="MTT_musicnn",
                input_length=self.sample_duration,
                input_overlap=False,
                extract_features=True,
            )

        embedding = np.array(features["mean_pool"]).flatten()

        return embedding


class MusicNNEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, sample_duration, sample_rate=16000):

        super().__init__(sample_duration=sample_duration, sample_rate=sample_rate)

        self.logger = logging.getLogger(__name__)

    def generate_embedding(self, audio_array: np.ndarray) -> np.ndarray:

        """Generate a musicnn max-pool and mean-pool embedding from an audio clip (represented as a 1D ndarray).

        :return: Embedding
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            _, _, features = extractor(
                audio=audio_array,
                sr=self.sample_rate,
                model="MTT_musicnn",
                input_length=self.sample_duration,
                input_overlap=False,
                extract_features=True,
            )

        embedding = np.concatenate(
            [
                np.array(features["mean_pool"]).flatten(),
                np.array(features["max_pool"]).flatten(),
            ]
        )

        return embedding
