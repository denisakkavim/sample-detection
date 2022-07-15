import logging
import os
import subprocess

from glob import glob
from typing import Optional, Iterable

from sample_detection.scrape.base import BaseScraper


class YoutubeScraper(BaseScraper):
    def __init__(self, save_dir):

        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.save_dir = save_dir

    def download_youtube_audio(
        self, youtube_id: str, file_name: Optional[str] = None
    ) -> bool:

        """Download audio from a given YouTube video.

        :param youtube_id: The ID of the YouTube video to download
        :type youtube_id: str
        :param save_dir: The directory to save the downloaded file in
        :type save_dir: str
        :param file_name: The name of the downloaded file. Defaults to the YouTube ID of the video being downloaded.
        :type file_name: str, optional

        :return: Whether the download was successful
        """

        if file_name is None:
            file_name = youtube_id

        url = f"http://youtube.com/watch?v={youtube_id}"

        self.logger.info(f"Downloading YouTube ID {youtube_id}.")

        try:
            command = f"yt-dlp -f 'ba' -x --audio-format mp3 {url} -o {self.save_dir}/{file_name}.mp3"
            subprocess.run(
                command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            )
            self.logger.info(f"YouTube ID {youtube_id} successfully downloaded.")
        except:
            # Sometimes, this throws an error for reasons I can't figure out. This is an extremely rare event (happens on
            # 0.5% of downloads), so we assume (again) that it happens completely at random, and isn't a problem when
            # collecting data.
            self.logger.warning(
                f"YouTube ID {youtube_id} could not be downloaded. This YouTube ID will be skipped."
            )
            return False

        return True

    def scrape(self, youtube_ids: Iterable[str]) -> bool:

        """Download the audio for the videos with the given YouTube IDs, and save them in save_dir.

        :param youtube_ids: IDs of YouTube videos to download the audio from
        :type youtube_ids: Iterable
        :param save_dir: Path to the folder in which the downloaded audio should be saved
        :type save_dir: str

        :return: True
        """

        # Check if any songs have already been downloaded, and if yes, exclude them from the set of songs to download:
        ids_already_downloaded = {
            self.extract_filename_from_filepath(path)
            for path in glob(os.path.join(self.save_dir, "*.mp3"))
        }
        ids_to_download = youtube_ids - ids_already_downloaded

        self.logger.info(f"Downloading audio from YouTube.")

        if ids_already_downloaded:
            self.logger.info(
                f'Some YouTube IDs have already been downloaded: {",".join(ids_already_downloaded)}'
            )
            self.logger.info("Downloading remaining YouTube IDs...")

        for id in ids_to_download:
            self.download_youtube_audio(youtube_id=id)

        return True
