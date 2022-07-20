import logging
import numpy as np
import wav2clip

from sample_detection.detect.embedding_generators.base import EmbeddingGenerator


class Wav2ClipEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, sample_duration, sample_rate=16000):

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
