import logging

from bs4 import BeautifulSoup
from time import sleep
from urllib.request import Request, urlopen
from urllib.error import URLError


class HTMLScraper:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_web_page(
        self, url: str, attempts: int = 5, wait_between_attempts: int = 10
    ) -> BeautifulSoup:
        """Gets the HTML content of a webpage.

        :param url: URL of the webpage to get content from
        :type url: str
        :param attempts: Number of attempts to reach the webpage
        :type attempts: int
        :param wait_between_attempts: Time (in seconds) to wait between attempts
        :type wait_between_attempts: int

        :rtype: bs4.BeautifulSoup

        :raises TimeoutError:
        :return: The HTML contents of the webpage at the given url, as a BeautifulSoup object.
        """

        # Check if we've exhausted all our attempts before trying to reach the page:
        if attempts == 0:
            raise URLError(
                f"Failed to get contents of web page. Is the internet connection down?"
            )

        self.logger.info(f"{attempts} attempts remaining: getting page at {url}")

        try:
            request = Request(
                url=url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Safari/605.1.15"
                },
            )
            response = urlopen(request, timeout=5)
            charset = response.headers.get_content_charset()
            content = response.read().decode(charset)
        except (URLError, TimeoutError) as err:
            self.logger.warning(f"Failed to get page at {url}")

            # If we get a URL error or a TimeoutError, wait before trying again:
            sleep(wait_between_attempts)
            content = self.get_web_page(
                url=url,
                attempts=attempts - 1,
                wait_between_attempts=wait_between_attempts,
            )

        return BeautifulSoup(markup=content, features="html.parser")
