import click
import logging
import os
import pickle

from sample_detection.detect.audio import Audio
from sample_detection.detect.sample_detector import SampleDetector
from sample_detection.scrape.load import load_sample_info
from sample_detection.scrape.scraper import SampleScraper


@click.group()
def cli():

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    return None


@cli.command()
@click.option("--threshold", default=1 - 1e-5)
@click.option("--hop-length", default=3)
@click.argument("sample_detector_path", type=click.Path())
@click.argument("audio_1_path", type=click.Path())
@click.argument("audio_2_path", type=click.Path())
def detect(sample_detector_path, audio_1_path, audio_2_path, threshold, hop_length):

    """Run sample-detection detect <SAMPLE_DETECTOR_PATH> <AUDIO_1_PATH> <AUDIO_2_PATH> --hop-length <HOP_LENGTH> --threshold <THRESHOLD>
    to detect whether audio_1 contains any samples of audio_2 (or vice versa).

    <SAMPLE_DETECTOR_PATH> is the path to a pre-trained sample_detector (can be generated with
    sample-detection train <INFO_PATH> <AUDIO_DIR> <SAVE_DIR> --sample-rate <SAMPLE_RATE> --sample-duration <SAMPLE_DURATION>).
    <AUDIO_1_PATH> is the path to the first audio file.
    <AUDIO_2_PATH> is the path to the second audio file

    <HOP_LENGTH> is the amount of time (in seconds) between the start of each potential sample.
    <THRESHOLD> is the value x such that if the score from the sample detector is greater than x, we show the sample to the user
    """

    with open(os.path.join(sample_detector_path, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    audio_1 = Audio(path=audio_1_path)
    audio_2 = Audio(path=audio_2_path)

    samples = model.find_samples(
        audio_1=audio_1, audio_2=audio_2, threshold=threshold, hop_length=hop_length
    )

    print(samples)

    return None


@cli.command()
@click.option("--start-year", default=2000)
@click.option("--end-year", default=2021)
@click.option("--pages-per-year", default=50)
@click.argument("save_dir", type=click.Path())
def scrape(save_dir, start_year, end_year, pages_per_year):

    """Run sample-detection scrape <SAVE_DIR> --start-year <START_YEAR> --end-year <END_YEAR> --pages-per-year <PAGES_PER_YEAR>
    to scrape WhoSampled for samples, and download the songs from YouTube.

    <START_YEAR> and <END_YEAR> refer to the release year of the song that contains the sample.
    <PAGES_PER_YEAR> dictates the number of pages to scrape on WhoSampled for each year.

    """

    scraper = SampleScraper(save_dir=save_dir)
    scraper.scrape(
        start_year=start_year, end_year=end_year, pages_per_year=pages_per_year
    )

    return None


@cli.command()
@click.option("--sample-rate", default=16000)
@click.option("--sample-duration", default=15)
@click.option("--min-negatives", default=1)
@click.argument("info_path", type=click.Path())
@click.argument("audio_dir", type=click.Path())
@click.argument("save_dir", type=click.Path())
def train(info_path, audio_dir, save_dir, sample_duration, sample_rate, min_negatives):

    """Run sample-detection train <INFO_PATH> <AUDIO_DIR> <SAVE_DIR> --sample-rate <SAMPLE_RATE> --sample-duration <SAMPLE_DURATION> --min-negatives <MIN_NEGATIVES>
    to train a model to detect samples.

    <INFO_PATH> is the path to sample information scraped from WhoSampled by the scraper.
    <AUDIO_DIR> is the directory containing the audio files scraped by the scraper.
    <SAVE_DIR> is the directory in which the resulting model should be saved.

    <SAMPLE_RATE> is the sample rate at which songs should be loaded (set to 16000 by default).
    <SAMPLE_DURATION> is the default length of a sample (defaults to 15 seconds).

    """

    train_df = load_sample_info(info_path)

    model = SampleDetector(sample_duration=sample_duration, sample_rate=sample_rate)
    model.fit(sample_info=train_df, audio_dir=audio_dir, min_negatives=min_negatives)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    return None
