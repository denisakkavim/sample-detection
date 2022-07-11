import click
import os
import pickle

from sample_detection.load.scraper import load_sample_info
from sample_detection.model.model import Model
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

    train_df = load_sample_info(info_path)

    model = Model(sample_duration=sample_duration, sample_rate=sample_rate)
    model.fit(sample_info=train_df, audio_dir=audio_dir, min_negatives=2)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    return None
