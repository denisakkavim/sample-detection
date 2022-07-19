# ROC AUC score: 0.7396896542787539
# Average Precision score: 0.7633839424755462

import click
import logging
import pickle

from sample_detection.detect.sample_detector import SampleDetector
from sample_detection.scrape.load import load_sample_info

from sklearn.metrics import roc_auc_score, average_precision_score


@click.command()
@click.option("--min-negatives", default=1)
@click.argument("val_info_path", type=click.Path())
@click.argument("model_path", type=click.Path())
@click.argument("audio_dir", type=click.Path())
def main(val_info_path, model_path, audio_dir, min_negatives):

    logger = logging.getLogger(__name__)

    eval_df = load_sample_info(val_info_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    (
        eval_df,
        embeddings,
    ) = model.embedding_generator.generate_embeddings_from_directory(
        sample_info=eval_df, audio_dir=audio_dir
    )

    negative_samples = model._generate_negative_samples(
        sample_info=eval_df, min_negatives=min_negatives
    )

    features, labels = model._format_embeddings(
        embeddings=embeddings,
        sample_info=eval_df,
        negative_samples=negative_samples,
    )

    scores = model.embedding_comparer.predict_proba(features)[:, 1]

    logger.info(f"ROC AUC score: {roc_auc_score(y_true = labels, y_score = scores)}")
    logger.info(
        f"Average Precision score: {average_precision_score(y_true = labels, y_score = scores)}"
    )

    logger.info("Done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
