import urllib.request

from sample_detection.scrape.whosampled import WhosampledScraper


def test_get_samples_on_page(
    whosampled_scraper, songs_in_sample_list, sample_list_html
):
    scraper_songs_in_sample_list = set(
        whosampled_scraper.get_samples_on_page(sample_list_html)
    )

    test_songs_in_sample_list = songs_in_sample_list

    assert scraper_songs_in_sample_list == test_songs_in_sample_list


def test_get_sample_details(
    monkeypatch, whosampled_scraper, sample_url, mock_response, sample_details
):
    def mock_urlopen(*args, **kwargs):
        return mock_response

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    scraped_sample_details = whosampled_scraper.get_sample_details(sample_url)
    test_sample_details = sample_details

    assert scraped_sample_details == test_sample_details
