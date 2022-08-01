# Wav2CLIP ROC AUC score: 0.7402642150002591
# Wav2CLIP Average Precision score: 0.757070016939695

# AudioCLIP ROC AUC score: 0.7350159384232623
# AudioCLIP Average Precision score: 0.7543083151217709

import click
import logging

from sample_detection.detect.embedding_generators.audioclip.generator import (
    AudioCLIPEmbeddingGenerator,
)
from sample_detection.detect.embedding_generators.wav2clip.generator import (
    Wav2ClipEmbeddingGenerator,
)
from sample_detection.detect.embedding_generators.musicnn.generator import (
    MusicNNEmbeddingGenerator,
    MusicNNMaxPoolEmbeddingGenerator,
    MusicNNMeanPoolEmbeddingGenerator,
)
from sample_detection.detect.sample_detector import SampleDetector
from sample_detection.scrape.load import load_sample_info

from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_embedding_generator(
    audio_dir,
    min_negatives,
    embedding_generator,
    train_df,
    eval_df,
    train_negative_samples=None,
    eval_negative_samples=None,
):

    return_samples = (train_negative_samples is None) and (
        eval_negative_samples is None
    )

    model = SampleDetector(
        sample_duration=15, sample_rate=44100, embedding_generator=embedding_generator
    )

    # train model

    (train_df, embeddings,) = embedding_generator.generate_embeddings_from_directory(
        sample_info=train_df, audio_dir=audio_dir
    )

    if train_negative_samples is None:
        train_negative_samples = model._generate_negative_samples(
            sample_info=train_df, min_negatives=min_negatives
        )

    features, labels = model._format_embeddings(
        embeddings=embeddings,
        sample_info=train_df,
        negative_samples=train_negative_samples,
    )

    model.embedding_comparer.fit(X=features, y=labels)

    # eval model

    (
        eval_df,
        eval_embeddings,
    ) = embedding_generator.generate_embeddings_from_directory(
        sample_info=eval_df, audio_dir=audio_dir
    )

    if eval_negative_samples is None:
        eval_negative_samples = model._generate_negative_samples(
            sample_info=eval_df, min_negatives=min_negatives
        )

    eval_features, eval_labels = model._format_embeddings(
        embeddings=eval_embeddings,
        sample_info=eval_df,
        negative_samples=eval_negative_samples,
    )

    eval_scores = model.embedding_comparer.predict_proba(eval_features)[:, 1]

    roc_auc = roc_auc_score(y_true=eval_labels, y_score=eval_scores)
    ap = average_precision_score(y_true=eval_labels, y_score=eval_scores)

    if return_samples:
        return roc_auc, ap, train_negative_samples, eval_negative_samples
    else:
        return roc_auc, ap


@click.command()
@click.option("--min-negatives", default=1)
@click.argument("val_info_path", type=click.Path())
@click.argument("train_info_path", type=click.Path())
@click.argument("audio_dir", type=click.Path())
def main(val_info_path, train_info_path, audio_dir, min_negatives):

    logger = logging.getLogger(__name__)

    train_df = load_sample_info(train_info_path)
    eval_df = load_sample_info(val_info_path)

    musicnn_max_emb_gen = MusicNNMaxPoolEmbeddingGenerator(sample_duration=15)

    (
        mnn_max_roc,
        mnn_max_ap,
        train_negative_samples,
        eval_negative_samples,
    ) = evaluate_embedding_generator(
        audio_dir=audio_dir,
        min_negatives=min_negatives,
        embedding_generator=musicnn_max_emb_gen,
        train_df=train_df,
        eval_df=eval_df,
    )

    musicnn_mean_emb_gen = MusicNNMeanPoolEmbeddingGenerator(sample_duration=15)

    (mnn_mean_roc, mnn_mean_ap,) = evaluate_embedding_generator(
        audio_dir=audio_dir,
        min_negatives=min_negatives,
        embedding_generator=musicnn_mean_emb_gen,
        train_df=train_df,
        eval_df=eval_df,
        train_negative_samples=train_negative_samples,
        eval_negative_samples=eval_negative_samples,
    )

    musicnn_cat_emb_gen = MusicNNEmbeddingGenerator(sample_duration=15)

    (mnn_cat_roc, mnn_cat_ap,) = evaluate_embedding_generator(
        audio_dir=audio_dir,
        min_negatives=min_negatives,
        embedding_generator=musicnn_cat_emb_gen,
        train_df=train_df,
        eval_df=eval_df,
        train_negative_samples=train_negative_samples,
        eval_negative_samples=eval_negative_samples,
    )

    w2c_emb_gen = Wav2ClipEmbeddingGenerator(sample_duration=15)

    (w2c_roc, w2c_ap,) = evaluate_embedding_generator(
        audio_dir=audio_dir,
        min_negatives=min_negatives,
        embedding_generator=w2c_emb_gen,
        train_df=train_df,
        eval_df=eval_df,
        train_negative_samples=train_negative_samples,
        eval_negative_samples=eval_negative_samples,
    )

    aclp_emb_gen = AudioCLIPEmbeddingGenerator(
        embedding_model_path="../models/embedding_generators/state_dict.pt",
        sample_duration=15,
    )

    (aclp_roc, aclp_ap,) = evaluate_embedding_generator(
        audio_dir=audio_dir,
        min_negatives=min_negatives,
        embedding_generator=aclp_emb_gen,
        train_df=train_df,
        eval_df=eval_df,
        train_negative_samples=train_negative_samples,
        eval_negative_samples=eval_negative_samples,
    )

    logger.info(f"MusicNN Max Pool AUC score: {mnn_max_roc}")
    logger.info(f"MusicNN Max Pool Average Precision score: {mnn_max_ap}")

    logger.info(f"MusicNN Mean Pool AUC score: {mnn_mean_roc}")
    logger.info(f"MusicNN Mean Pool Average Precision score: {mnn_mean_ap}")

    logger.info(f"MusicNN concat AUC score: {mnn_cat_roc}")
    logger.info(f"MusicNN concat Average Precision score: {mnn_cat_ap}")

    logger.info(f"W2C ROC AUC score: {w2c_roc}")
    logger.info(f"W2C Average Precision score: {w2c_ap}")

    logger.info(f"AudioCLIP ROC AUC score: {aclp_roc}")
    logger.info(f"AudioCLIP Average Precision score: {aclp_ap}")

    logger.info("Done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
