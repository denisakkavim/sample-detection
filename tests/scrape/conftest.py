import pytest

from bs4 import BeautifulSoup
from sample_detection.scrape.whosampled import WhosampledScraper


def html_to_bs4(html_filepath):
    with open(html_filepath, "r") as f:
        page = BeautifulSoup(f, features="html.parser")

    return page


@pytest.fixture
def whosampled_html_dir(test_file_dir):
    return test_file_dir / "whosampled_html"


@pytest.fixture
def sample_list_html(whosampled_html_dir):
    sample_list_filepath = whosampled_html_dir / "sample_list.html"
    return html_to_bs4(sample_list_filepath)


@pytest.fixture
def songs_in_sample_list():
    songs_in_sample_list = [
        "/sample/1007275/Metro-Boomin-The-Weeknd-21-Savage-Creepin%27-Young-Thug-Some-More/",
        "/sample/1006189/Coi-Leray-Players-Grandmaster-Flash-The-Furious-Five-Grandmaster-Melle-Mel-Duke-Bootee-The-Message/",
        "/sample/954838/Beyonc%C3%A9-BREAK-MY-SOUL-Big-Freedia-Explode/",
        "/sample/954830/Beyonc%C3%A9-BREAK-MY-SOUL-Robin-S.-Show-Me-Love-(Stonebridge-Mix)/",
        "/sample/998469/Drake-21-Savage-Rich-Flex-T.I.-24%27s/",
        "/sample/998690/Drake-21-Savage-Rich-Flex-Invasion-of-the-Bee-Girls-Nora%27s-Transformation/",
        "/sample/998466/Drake-21-Savage-Rich-Flex-Megan-Thee-Stallion-Savage/",
        "/sample/907254/The-Weeknd-Out-of-Time-Tomoko-Aran-Midnight-Pretenders/",
        "/sample/952685/Drake-21-Savage-Jimmy-Cooks-Playa-Fly-Just-Awaken-Shaken/",
        "/sample/953602/Drake-21-Savage-Jimmy-Cooks-Brook-Benton-You-Were-Gone/",
        "/sample/998526/Drake-21-Savage-Spin-Bout-U-Oobie-Give-Me-Your-Lovin/",
        "/sample/966113/Central-Cee-Doja-Eve-Gwen-Stefani-Let-Me-Blow-Ya-Mind/",
        "/sample/926803/James-Hype-Miggy-Dela-Rosa-Ferrari-P.-Diddy-Ginuwine-Loon-Mario-Winans-I-Need-a-Girl-(Part-2)/",
        "/sample/937937/Future-Drake-Tems-WAIT-FOR-U-Tems-Higher-(Live)/",
    ]

    return set(songs_in_sample_list)


@pytest.fixture
def sample_url():
    return "/sample/729975/Dua-Lipa-Love-Again-Lew-Stone-%26-the-Monseigneur-Band-Al-Bowlly-My-Woman/"


@pytest.fixture
def sample_details():
    sample_details = {
        "whosampled_id": "729975",
        "sample_in": "/Dua-Lipa/Love-Again/",
        "sample_from": "/Lew-Stone-%26-the-Monseigneur-Band/My-Woman/",
        "sample_in_times": [31],
        "sample_from_times": [0],
        "sample_in_ytid": "BC19kwABFwc",
        "sample_from_ytid": "PtJDq7f_EF0",
        "sample_type": "Direct Sample of Hook / Riff",
    }

    return sample_details


@pytest.fixture()
def mock_response(whosampled_html_dir):
    sample_html_filepath = whosampled_html_dir / "sample.html"

    class MockResponse:
        class MockHeader:
            @staticmethod
            def get_content_charset(self):
                return "utf-8"

        def __init__(self):
            self.headers = self.MockHeader()

        def read():
            return open(sample_html_filepath, "r")

    return MockResponse()


@pytest.fixture()
def whosampled_scraper():
    return WhosampledScraper()
