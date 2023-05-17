import logging
import os
import pandas as pd

from ffmpeg import probe
from glob import glob

from sample_detection.scrape.whosampled import WhosampledScraper
from sample_detection.scrape.youtube import (
    YoutubeAudioScraper,
    extract_filename_from_filepath,
)


class SampleScraper:
    def __init__(
        self,
        save_dir,
        attempts_per_page: int = 10,
        retry_after_seconds: int = 30,
        remove_audio_longer_than_seconds: int = 7 * 60,
    ):
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
            retry_after_seconds=retry_after_seconds,
        )
        self.youtube_scraper = YoutubeAudioScraper()
        self.AUDIO_LENGTH_REMOVAL_THRESHOLD = remove_audio_longer_than_seconds

    def _clean(self, sample_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(
            f"Identifying un-needed audio files (samples for which both songs could not be downloaded from YouTube or are over {self.AUDIO_LENGTH_REMOVAL_THRESHOLD} seconds long)"
        )

        # Get samples for which we have been able to download both songs from YouTube:
        all_downloaded_ids = {
            extract_filename_from_filepath(path)
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

    def scrape(self, start_year: int, end_year: int, pages_per_year: int) -> None:
        """Scrape WhoSampled for samples, and download the songs from YouTube.

        :param start_year: First year to get samples from
        :type start_year: int
        :param end_year: Last year to get samples from
        :type end_year: int
        :param pages_per_year: Number of pages of search results to go through for each year
        :type pages_per_year: int

        """

        self.logger.info("Starting WhoSampled scrape")
        sample_df = self.whosampled_scraper.get_samples_between_years(
            start_year, end_year, pages_per_year
        )
        self.logger.info("WhoSampled scrape completed")

        self.logger.info("Downloading audio for YouTube clips found in samples")
        ytids_to_download = set(sample_df["sample_in_ytid"]) | set(
            sample_df["sample_from_ytid"]
        )
        yt_scrape_success = self.youtube_scraper.download_audio_for_ids(
            youtube_ids=ytids_to_download, save_dir=self.save_dirs["audio"]
        )
        self.logger.info("YouTube download completed")

        self.logger.info("Saving sample details to disk")
        cleaned_df = self._clean(sample_df=sample_df)
        cleaned_df.to_csv(
            os.path.join(self.save_dirs["sample_details"], "sample_details.csv"),
            index=False,
        )
        self.logger.info("Sample details saved to disk")
