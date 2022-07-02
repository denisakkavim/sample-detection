import logging
import os
import pandas as pd

from ffmpeg import probe
from glob import glob

from sample_detection.scraper.base import BaseScraper
from sample_detection.scraper.whosampled import WhosampledScraper
from sample_detection.scraper.youtube import YoutubeScraper


class SampleScraper(BaseScraper):
    def __init__(
        self,
        save_dir,
        attempts_per_page: int = 10,
        wait_between_attempts: int = 30,
        audio_length_removal_threshold: int = 7 * 60,
    ):

        super().__init__()

        self.logger = logging.getLogger(__name__)

        self.save_dirs = {
            "root": save_dir,
            "sample_details": os.path.join(save_dir, "sample_details"),
            "audio": os.path.join(save_dir, "audio"),
        }
        for path in self.save_dirs.values():
            if not os.path.exists(path):
                os.makedirs(path)

        self.whosampled_scraper = WhosampledScraper(
            attempts_per_page=attempts_per_page,
            wait_between_attempts=wait_between_attempts,
        )
        self.youtube_scraper = YoutubeScraper(save_dir=self.save_dirs["audio"])
        self.AUDIO_LENGTH_REMOVAL_THRESHOLD = audio_length_removal_threshold

    def clean(self, sample_df: pd.DataFrame) -> pd.DataFrame:

        self.logger.info(
            f"Identifying un-needed audio files (samples for which both songs could not be downloaded from YouTube or are over {self.AUDIO_LENGTH_REMOVAL_THRESHOLD} seconds long)"
        )

        # Get samples for which we have been able to download both songs from YouTube:
        all_downloaded_ids = {
            self.extract_filename_from_filepath(path)
            for path in glob(os.path.join(self.save_dirs["audio"], "*.mp3"))
        }
        samples_to_keep = sample_df["sample_in_ytid"].isin(
            all_downloaded_ids
        ) & sample_df["sample_from_ytid"].isin(all_downloaded_ids)
        sample_df = sample_df[samples_to_keep]

        # Get samples in which both songs are less than AUDIO_LENGTH_REMOVAL_THRESHOLD seconds long:
        audio_lengths = {
            id: float(
                probe(os.path.join(self.save_dirs["audio"], id + ".mp3"))["format"][
                    "duration"
                ]
            )
            for id in all_downloaded_ids
        }
        ids_to_keep = {
            id
            for id, duration in audio_lengths.items()
            if duration < self.AUDIO_LENGTH_REMOVAL_THRESHOLD
        }
        samples_to_keep = sample_df["sample_in_ytid"].isin(ids_to_keep) & sample_df[
            "sample_from_ytid"
        ].isin(ids_to_keep)
        sample_df = sample_df[samples_to_keep]

        # Some audio will have been downloaded that is over the removal threshold, or the other song in the sample could not be downloaded.
        # These aren't needed, so delete them:

        self.logger.info("Deleting un-needed audio files")
        youtube_ids_needed = set(sample_df["sample_in_ytid"]) | set(
            sample_df["sample_from_ytid"]
        )
        ids_to_delete = all_downloaded_ids - youtube_ids_needed
        for id in ids_to_delete:
            os.remove(os.path.join(self.save_dirs["audio"], f"{id}.mp3"))
            self.logger.info(f"{id}.mp3 deleted")

        return sample_df

    def scrape(self, start_year, end_year, pages_per_year):

        self.logger.info("Starting WhoSampled scrape")
        sample_df = self.whosampled_scraper.scrape(start_year, end_year, pages_per_year)
        self.logger.info("WhoSampled scrape completed")

        self.logger.info("Downloading audio for YouTube clips found in samples")
        ytids_to_download = set(sample_df["sample_in_ytid"]) | set(
            sample_df["sample_from_ytid"]
        )
        yt_scrape_success = self.youtube_scraper.scrape(youtube_ids=ytids_to_download)
        self.logger.info("YouTube download completed")

        self.logger.info("Saving sample details to disk")
        cleaned_df = self.clean(sample_df=sample_df)
        cleaned_df.to_csv(
            os.path.join(self.save_dirs["sample_details"], "sample_details.csv"),
            index=False,
        )
        self.logger.info("Sample details saved to disk")
