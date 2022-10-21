import logging
import numpy as np
import pandas as pd
import random
import torch

from typing import Dict, List, Optional, Set, Tuple

from sample_detection.detect.audio import Audio
from sample_detection.detect.embedding_generators.base import EmbeddingGenerator
from sample_detection.detect.embedding_generators.wav2clip.generator import (
    Wav2ClipEmbeddingGenerator,
)
from sample_detection.detect.mlp import MLPClassifier


class SampleDetector:
    def __init__(
        self,
        sample_duration: int,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        learning_rate: float = 1e-3,
        dropout: float = 0.0,
        batch_size: int = 32,
        hidden_layer_sizes: Tuple[int, int] = (256, 64),
        max_iter: int = int(1e7),
        random_state: int = 42,
    ):

        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True
        self.random_state = random_state
        self.sample_duration = sample_duration

        self.logger.info(
            f"Setting random seed to {random_state} for base python, numpy, and torch"
        )
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        random.seed(random_state)

        if embedding_generator is None:
            self.embedding_generator = Wav2ClipEmbeddingGenerator(
                sample_duration=sample_duration
            )
        else:
            self.embedding_generator = embedding_generator

        self.embedding_comparer = MLPClassifier(
            dropout=dropout,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            tol=1e-4,
        )

    def _generate_negative_samples(
        self, sample_info: pd.DataFrame, min_negatives: Optional[int] = 1
    ) -> pd.DataFrame:

        """Given a dataframe of samples, use negative sampling to generate sets
        of negative pairs (i.e: pairs of songs that do not contain a sample).

        :param sample_df: Dataframe with sample info
        :type sample_df: pd.DataFrame

        :param min_negatives: Minimum number of pairs per positive pair.
        :type min_negatives: Optional[int]

        :return: Dataframe containing positive/negative sample pairs
        :rtype: pd.DataFrame

        """

        def generate_group_pairs(
            group_id: str,
            group_df: pd.DataFrame,
            available_song_ids: Set[str],
            min_negatives: int = 1,
        ) -> pd.DataFrame:

            negatives = available_song_ids - (
                set(group_df["sample_in_ytid"]) | set(group_df["sample_from_ytid"])
            )

            n_negatives = max(min(min_negatives, len(negatives)), len(group_df))
            negatives = random.sample(list(negatives), n_negatives)
            negative_df = pd.DataFrame({"sample_from_ytid": negatives})
            negative_df["sample_in_ytid"] = group_id

            return negative_df

        available_song_ids = set(sample_info["sample_in_ytid"]) | set(
            sample_info["sample_from_ytid"]
        )

        positive_df = sample_info.loc[:, ["sample_from_ytid", "sample_in_ytid"]]

        sample_used_in_groups = positive_df.groupby(["sample_in_ytid"])
        negative_df = pd.concat(
            [
                generate_group_pairs(
                    group_id=id,
                    group_df=df,
                    available_song_ids=available_song_ids,
                    min_negatives=min_negatives,
                )
                for id, df in sample_used_in_groups
            ]
        )

        return negative_df

    def _process_positive_samples(
        self, embeddings: Dict[str, Dict[int, np.ndarray]], sample_info: pd.DataFrame
    ) -> List[Tuple[np.ndarray, int]]:

        feature_list = []

        for _, row in sample_info.iterrows():

            sample_in_id = row["sample_in_ytid"]
            sample_in_times = row["sample_in_times"]

            sample_from_id = row["sample_from_ytid"]
            sample_from_times = row["sample_from_times"]

            NUM_SAMPLE_INSTANCES = len(sample_in_times)

            for i in range(NUM_SAMPLE_INSTANCES):

                embedding_sample_from = embeddings[sample_from_id][sample_from_times[i]]
                embedding_sample_used_in = embeddings[sample_in_id][sample_in_times[i]]

                features = np.concatenate(
                    (embedding_sample_from, embedding_sample_used_in)
                )
                feature_list.append((features, 1))

                features = np.concatenate(
                    (embedding_sample_used_in, embedding_sample_from)
                )
                feature_list.append((features, 1))

        return feature_list

    def _process_negative_samples(
        self,
        embeddings: Dict[str, Dict[int, np.ndarray]],
        negative_samples: pd.DataFrame,
    ) -> List[Tuple[np.ndarray, int]]:

        feature_list = []

        for _, row in negative_samples.iterrows():

            sample_in_id = row["sample_in_ytid"]
            sample_from_id = row["sample_from_ytid"]

            embedding_sample_from = random.choice(
                list(embeddings[sample_from_id].values())
            )
            embedding_sample_used_in = random.choice(
                list(embeddings[sample_in_id].values())
            )

            features = np.concatenate((embedding_sample_from, embedding_sample_used_in))
            feature_list.append((features, 0))

            features = np.concatenate((embedding_sample_used_in, embedding_sample_from))
            feature_list.append((features, 0))

        return feature_list

    def _format_embeddings(
        self,
        embeddings: Dict[str, Dict[int, np.array]],
        sample_info: pd.DataFrame,
        negative_samples: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:

        positive_samples = self._process_positive_samples(
            embeddings=embeddings, sample_info=sample_info
        )

        negative_samples = self._process_negative_samples(
            embeddings=embeddings, negative_samples=negative_samples
        )

        # join positive and negative samples together, and separate features from labels
        features, labels = zip(*(positive_samples + negative_samples))

        return np.vstack(features), np.array(labels)

    def fit(
        self, sample_info: pd.DataFrame, audio_dir: str, min_negatives: int = 1
    ) -> None:

        """Fit the sample detection model.

        :param sample_info: Sample info (as scraped from scraper), with sample times as lists
        :type sample_info: pd.DataFrame
        :param audio_dir: Directory containing the audio files scraped by the scraper.
        :type audio_dir: str
        :param min_negatives: Number of negative samples to generate per positive sample
        :type min_negatives: pd.DataFrame

        """

        (
            sample_info,
            embeddings,
        ) = self.embedding_generator.generate_embeddings_from_directory(
            sample_info=sample_info, audio_dir=audio_dir
        )

        negative_samples = self._generate_negative_samples(
            sample_info=sample_info, min_negatives=min_negatives
        )

        features, labels = self._format_embeddings(
            embeddings=embeddings,
            sample_info=sample_info,
            negative_samples=negative_samples,
        )

        self.embedding_comparer.fit(X=features, y=labels)

    def predict(
        self,
        audio_1: Optional[Audio] = None,
        audio_2: Optional[Audio] = None,
        embedding_1: Optional[np.ndarray] = None,
        embedding_2: Optional[np.ndarray] = None,
    ):

        """Predict whether an audioclip contains a sample of another. All inputs are assumed to be audioclips (
        or embeddings of audioclips) of length sample_duration (specified in constructor, and can be found
        under the sample_duration attribute).

        :return: Score indicating likelihood that one audioclip contains a sample of another
        """

        POSITIVE_CLASS_INDEX = 1
        RESULT_INDEX = 0
        AUDIO_AND_EMBEDDING_MSG = "Both audio and an embedding have been passed in for at least one sound clip. The embedding(s) will be used."

        warn_audio_and_embedding = (
            audio_1 is not None and embedding_1 is not None
        ) or (audio_2 is not None and embedding_2 is not None)

        if warn_audio_and_embedding:
            self.logger.warning(AUDIO_AND_EMBEDDING_MSG)

        if embedding_1 is None:
            audio_1 = audio_1.get_numpy()
            embedding_1 = self.embedding_generator.generate_embedding(audio_1)

        if embedding_2 is None:
            audio_2 = audio_2.get_numpy()
            embedding_2 = self.embedding_generator.generate_embedding(audio_2)

        features = np.concatenate((embedding_1, embedding_2)).reshape(1, -1)
        score = self.embedding_comparer.predict_proba(X=features)[
            :, POSITIVE_CLASS_INDEX
        ][RESULT_INDEX]

        return score

    def find_samples(
        self, audio_1: Audio, audio_2: Audio, threshold: float, hop_length: int = 3
    ) -> List[Dict[str, float]]:

        """Given two songs, find instances where one song contains a sample of the other.

        :param audio_1: Song 1
        :type audio_1: Audio
        :param audio_1: Song 2
        :type audio_2: Audio
        :param threshold: Threshold above which to return samples.
        :type audio_2: float

        :return: List of dictionaries. Each dictionary contains the start time of the sample
        in each song, the length of the sample, and the confidence score generated by the
        model for the sample.
        """

        audio_1_embeddings = {
            start_1: self.embedding_generator.generate_embedding(
                audio_1.get_extract(
                    start_time=start_1, extract_length=self.sample_duration
                ).get_numpy()
            )
            for start_1 in range(0, len(audio_1), hop_length)
        }

        audio_2_embeddings = {
            start_2: self.embedding_generator.generate_embedding(
                audio_2.get_extract(
                    start_time=start_2, extract_length=self.sample_duration
                ).get_numpy()
            )
            for start_2 in range(0, len(audio_2), hop_length)
        }

        potential_samples = [
            {
                "start_time_1": start_1,
                "start_time_2": start_2,
                "sample_duration": self.sample_duration,
                "confidence": self.predict(
                    embedding_1=embedding_1, embedding_2=embedding_2
                ),
            }
            for start_1, embedding_1 in audio_1_embeddings.items()
            for start_2, embedding_2 in audio_2_embeddings.items()
        ]

        return [
            sample for sample in potential_samples if sample["confidence"] > threshold
        ]
