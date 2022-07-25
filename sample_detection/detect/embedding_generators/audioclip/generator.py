import logging
import numpy as np
import os
import torch


from sample_detection.detect.embedding_generators.base import EmbeddingGenerator
from sample_detection.detect.embedding_generators.audioclip._model.fbsp import (
    ESResNeXtFBSP,
)


class AudioCLIPEmbeddingGenerator(EmbeddingGenerator):
    def __init__(
        self, embedding_model_path, sample_duration: int = 15, sample_rate: int = 44100
    ):

        super().__init__(sample_duration=sample_duration, sample_rate=sample_rate)

        self.logger = logging.getLogger(__name__)
        self.model = ESResNeXtFBSP(
            n_fft=2048,
            hop_length=561,
            win_length=1654,
            window="blackmanharris",
            normalized=True,
            onesided=True,
            spec_height=-1,
            spec_width=-1,
            num_classes=1024,
            apply_attention=True,
            pretrained=False,
        )
        self.model.load_state_dict(torch.load(embedding_model_path))
        self.model.eval()

    def generate_embedding(self, audio_array):

        with torch.no_grad():
            audio_array = torch.tensor(audio_array.reshape(1, -1))

            embedding = self.model(audio_array)
            embedding = embedding / torch.linalg.norm(embedding, dim=-1, keepdim=True)
            embedding = np.array(embedding).flatten()

        return embedding
