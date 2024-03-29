import logging
import os
import subprocess

from glob import glob
from pathlib import Path
from typing import Optional, Iterable, Union


def extract_filename_from_filepath(path: str) -> str:
    """Given a path to a file, extract the name of the file without the extension.

    :param path: path to file
    :type path: str

    :return: filename of file in path
    :rtype: str
    """

    ID_SPLIT_INDEX = 0

    filename = os.path.basename(path).split(".")[ID_SPLIT_INDEX]

    return filename


class YoutubeAudioScraper:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _scrape_audio(
        self,
        youtube_id: str,
        save_dir: Union[Path, str],
        file_name: Optional[str] = None,
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
            command = f"yt-dlp -f 'ba' -x --audio-format mp3 {url} -o {save_dir}/{file_name}.mp3"
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

    def download_audio_for_ids(
        self, youtube_ids: Iterable[str], save_dir: Union[Path, str]
    ) -> bool:
        """Download the audio for the videos with the given YouTube IDs, and save them in save_dir.

        :param youtube_ids: IDs of YouTube videos to download the audio from
        :type youtube_ids: Iterable
        :param save_dir: Path to the folder in which the downloaded audio should be saved
        :type save_dir: str

        :return: True
        """

        # Check if any songs have already been downloaded, and if yes, exclude them from the set of songs to download:
        ids_already_downloaded = {
            extract_filename_from_filepath(path)
            for path in glob(os.path.join(save_dir, "*.mp3"))
        }
        ids_to_download = youtube_ids - ids_already_downloaded

        self.logger.info(f"Downloading audio from YouTube.")

        if ids_already_downloaded:
            self.logger.info(
                f'Some YouTube IDs have already been downloaded: {",".join(ids_already_downloaded)}'
            )
            self.logger.info("Downloading remaining YouTube IDs...")

        for id in ids_to_download:
            self._scrape_audio(youtube_id=id)

        return True
