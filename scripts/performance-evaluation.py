# Wav2CLIP ROC AUC score: 0.7402642150002591
# Wav2CLIP Average Precision score: 0.757070016939695

# AudioCLIP ROC AUC score: 0.7350159384232623
# AudioCLIP Average Precision score: 0.7543083151217709

import click
import logging
import os
import pickle

from sample_detection.detect.embedding_generators.audioclip.generator import (
    AudioCLIPEmbeddingGenerator,
)
from sample_detection.detect.embedding_generators.wav2clip.generator import (
    Wav2ClipEmbeddingGenerator,
)
from sample_detection.detect.sample_detector import SampleDetector
from sample_detection.scrape.load import load_sample_info

from sklearn.metrics import roc_auc_score, average_precision_score


@click.command()
@click.option("--min-negatives", default=1)
@click.argument("val_info_path", type=click.Path())
@click.argument("train_info_path", type=click.Path())
@click.argument("audio_dir", type=click.Path())
def main(val_info_path, train_info_path, audio_dir, min_negatives):

    logger = logging.getLogger(__name__)

    train_df = load_sample_info(train_info_path)
    eval_df = load_sample_info(val_info_path)

    aclp_emb_gen = AudioCLIPEmbeddingGenerator(
        embedding_model_path="../models/embedding_generators/state_dict.pt",
        sample_duration=15,
    )

    model = SampleDetector(
        sample_duration=15, sample_rate=44100, embedding_generator=aclp_emb_gen
    )

    # train model

    (train_df, embeddings,) = aclp_emb_gen.generate_embeddings_from_directory(
        sample_info=train_df, audio_dir=audio_dir
    )

    negative_samples = model._generate_negative_samples(
        sample_info=train_df, min_negatives=min_negatives
    )

    features, labels = model._format_embeddings(
        embeddings=embeddings,
        sample_info=train_df,
        negative_samples=negative_samples,
    )

    model.embedding_comparer.fit(X=features, y=labels)

    if not os.path.exists("../models/train/audioclip/"):
        os.makedirs("../models/train/audioclip/")

    with open("../models/train/audioclip/train.pkl", "wb") as f:
        pickle.dump(model, f)

    # eval model

    (eval_df, eval_embeddings,) = aclp_emb_gen.generate_embeddings_from_directory(
        sample_info=eval_df, audio_dir=audio_dir
    )

    eval_negative_samples = model._generate_negative_samples(
        sample_info=eval_df, min_negatives=min_negatives
    )

    eval_features, eval_labels = model._format_embeddings(
        embeddings=eval_embeddings,
        sample_info=eval_df,
        negative_samples=eval_negative_samples,
    )

    eval_scores = model.embedding_comparer.predict_proba(eval_features)[:, 1]

    audioclip_roc_auc = roc_auc_score(y_true=eval_labels, y_score=eval_scores)
    audioclip_ap = average_precision_score(y_true=eval_labels, y_score=eval_scores)

    logger.info(f"AudioCLIP ROC AUC score: {audioclip_roc_auc}")
    logger.info(f"AudioCLIP Average Precision score: {audioclip_ap}")

    w2c_emb_gen = Wav2ClipEmbeddingGenerator(sample_duration=15)

    model = SampleDetector(
        sample_duration=15, sample_rate=16000, embedding_generator=w2c_emb_gen
    )

    # train model

    (train_df, embeddings,) = w2c_emb_gen.generate_embeddings_from_directory(
        sample_info=train_df, audio_dir=audio_dir
    )

    features, labels = model._format_embeddings(
        embeddings=embeddings,
        sample_info=train_df,
        negative_samples=negative_samples,  # we should re-use the existing negative samples
    )

    model.embedding_comparer.fit(X=features, y=labels)

    if not os.path.exists("../models/train/wav2clip/"):
        os.makedirs("../models/train/wav2clip/")

    with open("../models/train/wav2clip/train.pkl", "wb") as f:
        pickle.dump(model, f)

    # eval model

    (eval_df, eval_embeddings,) = w2c_emb_gen.generate_embeddings_from_directory(
        sample_info=eval_df, audio_dir=audio_dir
    )

    eval_features, eval_labels = model._format_embeddings(
        embeddings=eval_embeddings,
        sample_info=eval_df,
        negative_samples=eval_negative_samples,  # we should re-use the existing negative samples
    )

    eval_scores = model.embedding_comparer.predict_proba(eval_features)[:, 1]

    w2c_roc_auc = roc_auc_score(y_true=eval_labels, y_score=eval_scores)
    w2c_ap = average_precision_score(y_true=eval_labels, y_score=eval_scores)

    logger.info(f"W2C ROC AUC score: {w2c_roc_auc}")
    logger.info(f"W2C Average Precision score: {w2c_ap}")
    logger.info(f"AudioCLIP ROC AUC score: {audioclip_roc_auc}")
    logger.info(f"AudioCLIP Average Precision score: {audioclip_ap}")

    logger.info("Done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
