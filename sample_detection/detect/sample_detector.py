import librosa
import numpy as np

from sample_detection.model.model import Model
from sample_detection.model.audio import Audio


class SampleDetector:
    def __init__(self, hop_length: int = 5, model: Model = None):
        self.model = model
        self.hop_length = hop_length
        self.sample_duration = self.model.sample_duration

    def find_samples(self, audio_1: Audio, audio_2: Audio, threshold: float):

        audio_1_embeddings = {
            start_1: self.model.embedding_generator.generate_embedding(
                audio_1.get_extract(
                    start_time=start_1, extract_length=self.sample_duration
                ).get_numpy()
            )
            for start_1 in range(0, len(audio_1), self.hop_length)
        }

        audio_2_embeddings = {
            start_2: self.model.embedding_generator.generate_embedding(
                audio_2.get_extract(
                    start_time=start_2, extract_length=self.sample_duration
                ).get_numpy()
            )
            for start_2 in range(0, len(audio_2), self.hop_length)
        }

        potential_samples = [
            {
                "time_song_1": (
                    start_song_1,
                    start_song_1 + self.sample_duration,
                ),
                "time_song_2": (
                    start_song_2,
                    start_song_2 + self.sample_duration,
                ),
                "confidence": self.model.predict(
                    embedding_1=embedding_1, embedding_2=embedding_2
                ),
            }
            for start_song_1, embedding_1 in audio_1_embeddings.items()
            for start_song_2, embedding_2 in audio_2_embeddings.items()
        ]

        return [
            sample for sample in potential_samples if sample["confidence"] > threshold
        ]
