import logging
import re
import pandas as pd

from bs4 import BeautifulSoup
from itertools import product
from typing import Dict, List
from urllib.error import URLError
from urllib.parse import urljoin

from sample_detection.scrape.base import BaseScraper


class WhosampledScraper(BaseScraper):
    def __init__(self, attempts_per_page: int = 10, wait_between_attempts: int = 30):
        """Create a WhoSampled Scraper.

        :param attempts_per_page: Number of attempts that should be made to load a given page
        :type attempts_per_page: int
        :param wait_between_attempts: Number of seconds to wait before re-attempting to load a page
        :type wait_between_attempts: Model

        """

        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.base_url = "http://173.192.193.226"
        self.attempts_per_page = attempts_per_page
        self.wait_between_attempts = wait_between_attempts

    def get_samples_on_page(self, page: BeautifulSoup) -> List[str]:
        """Get the URLs of all samples on the a given WhoSampled page.

        :param page: HTML contents of a page on WhoSampled, as returned by sample_detection.scraper.get_web_page.
        :type page: bs4.BeautifulSoup

        :return: List of strings. Each string is the URL of a sample on WhoSampled, relative to whosampled.com (e.g: if the full URL would be
        https://www.whosampled.com/sample/729975/Dua-Lipa-Love-Again-Lew-Stone-%26-the-Monseigneur-Band-Al-Bowlly-My-Woman/, the string in the list
        would be /sample/729975/Dua-Lipa-Love-Again-Lew-Stone-%26-the-Monseigneur-Band-Al-Bowlly-My-Woman/).

        """

        songs_on_page = page.find(
            name="section", attrs={"class": "trackList bordered-list"}
        ).find_all(name="article", attrs={"class": "trackItem"})

        # Initialise empty list to add to once we've got the samples on the current page:
        sample_urls_on_page = []

        for song in songs_on_page:
            try:
                sampled_songs = song.find(
                    name="span",
                    attrs={"class": "sampleAction"},
                    string=re.compile("sampled\n"),
                ).parent.find_all(name="a", attrs={"class": "connectionName playIcon"})
            except AttributeError:
                # If there are no sampled songs on the current page (may happen if a song is sampled far more often than it uses samples),
                # we need to explicitly return an empty list, as the parent used above won't exist:
                sampled_songs = []

            sample_urls_on_page = sample_urls_on_page + [
                sample.get("href") for sample in sampled_songs
            ]

        return sample_urls_on_page

    def get_sample_details(self, sample_url: str) -> Dict[str, str]:
        """Get the details of a sample: the WhoSampled sample ID (scraped from the URL), links to (on WhoSampled) the songs in the sample, YouTube
        IDs for the songs on YouTube (assuming they exist), and the type of sample.

        :param sample_url: URL for a sample, relative to whosampled.com (as returned by sample_detection.scraper.whosampled.get_samples_on_page)
        :type sample_url: str

        :return: Dict containing details of the sample at the given URL
        """

        try:
            absolute_url = urljoin(self.base_url, sample_url)
            sample_page = self.get_web_page(
                absolute_url,
                attempts=self.attempts_per_page,
                wait_between_attempts=self.wait_between_attempts,
            )
            sample_boxes = sample_page.find_all(
                name="div", attrs={"class": "sampleEntryBox"}
            )

            self.logger.info(f"Extracting sample details from sample {sample_url}")

            # sample_boxes is a list of beautiful soup objects - these indices are where the relevant information is stored:
            BOX_SAMPLE_USED_IN = 0
            BOX_SAMPLE_FROM = 1

            # not related to sample_boxes - this index tells us where to look to get the sample ID after splitting the sample URL
            SAMPLE_ID_SPLIT_INDEX = 2

            sample_details = {
                "whosampled_id": sample_url.split("/")[SAMPLE_ID_SPLIT_INDEX],
                "sample_in": sample_boxes[BOX_SAMPLE_USED_IN].a["href"],
                "sample_from": sample_boxes[BOX_SAMPLE_FROM].a["href"],
                "sample_in_times": [
                    int(time)
                    for time in sample_boxes[BOX_SAMPLE_USED_IN]
                    .find(name="strong", attrs={"id": "sample-dest-timing"})[
                        "data-timings"
                    ]
                    .split(",")
                ],
                "sample_from_times": [
                    int(time)
                    for time in sample_boxes[BOX_SAMPLE_FROM]
                    .find(name="strong", attrs={"id": "sample-source-timing"})[
                        "data-timings"
                    ]
                    .split(",")
                ],
                "sample_in_ytid": sample_boxes[BOX_SAMPLE_USED_IN].find(
                    name="div", attrs={"class": "embed-placeholder youtube-placeholder"}
                )["data-id"],
                "sample_from_ytid": sample_boxes[BOX_SAMPLE_FROM].find(
                    name="div", attrs={"class": "embed-placeholder youtube-placeholder"}
                )["data-id"],
                "sample_type": sample_page.find(
                    name="h2", attrs={"class": "section-header-title"}
                ).text,
            }

            self.logger.info(
                f"Successfully extracted sample details from sample {sample_url}"
            )

            return sample_details

        except Exception:
            # This is lazy of me, but there are so many things that could wrong when scraping any single WhoSampled page. Thankfully,
            # these don't happen very often. For now, I'm explicitly assuming this happens completely at random (this seems to be the case),
            # and so it's fine to pass when an error comes up (this is intended as a catch all against these errors). Best practice would be
            # to throw exceptions if/when an error comes up and handle it downstream, but I'm unlikely to get around to this, since it works
            # well enough as it stands.

            # TODO: Add exceptions for the various things that might go wrong when calling this (e.g: youtube link does not exist, get_page
            # fails, etc...)

            self.logger.warning(
                f"Failed to extract sample details from sample {sample_url}. This sample will be skipped."
            )

            pass

    def scrape(
        self, start_year: int = 2010, end_year: int = 2020, pages_per_year: int = 50
    ) -> pd.DataFrame:
        """Scrape WhoSample for samples in between (and including) the start and end year. Goes through pages_per_year
        pages of search results for each year.

        :param start_year: First year to get samples from
        :type start_year: int
        :param end_year: Last year to get samples from
        :type end_year: int
        :param pages_per_year: Number of pages of search results to go through for each year
        :type pages_per_year: int

        :return: Dataframe containing sample details as gotten from sample_detection.scraper.whosampled.get_sample_details

        """
        # Get tuples of (year, page) to scrape over (this makes the for loop below much more elegant):
        pages_to_scrape = list(
            product(range(start_year, end_year + 1), range(1, pages_per_year + 1))
        )

        # Initialise empty list to add to as we get sample details:
        sample_details = []

        self.logger.info("Getting sample details from WhoSampled.")

        for year, browse_page in pages_to_scrape:
            try:
                browse_page_url = (
                    f"{self.base_url}/browse/year/{year}/samples/{browse_page}/"
                )
                browse_page = self.get_web_page(
                    url=browse_page_url,
                    attempts=self.attempts_per_page,
                    wait_between_attempts=self.wait_between_attempts,
                )
                sample_urls_on_page = self.get_samples_on_page(page=browse_page)
                sample_details = sample_details + [
                    self.get_sample_details(sample_url=url)
                    for url in sample_urls_on_page
                ]
            except Exception as e:
                if isinstance(e, URLError):
                    raise e
                else:
                    pass

        # If we pass in the loop above, the corresponding entry in the list will be None - we need to filter these
        # out before we can create a DataFrame of what we've scraped:
        sample_details = [sample for sample in sample_details if sample is not None]

        return pd.DataFrame(sample_details)
