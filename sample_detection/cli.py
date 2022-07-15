import click
import os
import pickle

from sample_detection.load import load_sample_info
from sample_detection.detect.sample_detector import SampleDetector
from sample_detection.scraper.scraper import SampleScraper


@click.group()
def cli():
    pass


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
@click.argument("info_path", type=click.Path())
@click.argument("audio_dir", type=click.Path())
@click.argument("save_dir", type=click.Path())
def train(info_path, audio_dir, save_dir, sample_duration, sample_rate):

    """Run sample-detection train <INFO_PATH> <AUDIO_DIR> <SAVE_DIR> --sample-rate <SAMPLE_RATE> --sample-duration <SAMPLE_DURATION>
    to train a model to detect samples.

    <INFO_PATH> is the path to sample information scraped from WhoSampled by the scraper.
    <AUDIO_DIR> is the directory containing the audio files scraped by the scraper.
    <SAVE_DIR> is the directory in which the resulting model should be saved.

    <SAMPLE_RATE> is the sample rate at which songs should be loaded (set to 16000 by default).
    <SAMPLE_DURATION> is the default length of a sample (defaults to 15 seconds).

    """

    train_df = load_sample_info(info_path)

    model = SampleDetector(sample_duration=sample_duration, sample_rate=sample_rate)
    model.fit(sample_info=train_df, audio_dir=audio_dir, min_negatives=2)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    return None
