import logging
import numpy as np
import pandas as pd
import random
import torch

from sklearn.neural_network import MLPClassifier
from typing import Dict, List, Optional, Set, Tuple

from sample_detection.model.embedding_generators import Wav2ClipEmbeddingGenerator
from sample_detection.model.audio import Audio


class Model:
    def __init__(
        self,
        sample_duration: int,
        sample_rate: int,
        learning_rate: str = "constant",
        learning_rate_init: float = 1e-3,
        hidden_layer_sizes: Optional[Tuple[int, int]] = (256, 64),
        activation: Optional[str] = "relu",
        max_iter: Optional[int] = int(1e7),
        random_state: Optional[int] = 42,
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

        self.embedding_generator = Wav2ClipEmbeddingGenerator(
            sample_duration=sample_duration, sample_rate=sample_rate
        )
        self.embedding_comparer = MLPClassifier(
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            max_iter=max_iter,
            random_state=random_state,
        )

    def _generate_negative_samples(
        self, sample_info: pd.DataFrame, min_negatives: Optional[int] = 1
    ) -> pd.DataFrame:

        """Given a dataframe of samples, generate sets of negative pairs (i.e: pairs of songs
        that do not contain a sample).

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
            min_negatives: Optional[int] = 1,
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

        features, labels = zip(*(positive_samples + negative_samples))

        return np.vstack(features), np.array(labels)

    def fit(
        self, sample_info: pd.DataFrame, audio_dir: str, min_negatives: int
    ) -> None:

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
