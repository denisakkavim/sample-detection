Sample Detection
==============================

In music, sampling is the reuse of a portion (or sample) of a sound recording in another recording. Sample Detection is a Python library that allows users to detect whether one song contains a direct sample of another. 

This library has two components: 
- a scraper, to scrape WhoSampled for known samples and get the audio from YouTube
- a model builder, to build machine learning models to detect whether one song contains a sample of another

Organization
------------

    ├── .github
    │   └── workflows                     <- automated tests on merge/push to master   
    ├── docs                              <- sphinx docs
    ├── models                            <- place to store trained models
    ├── scripts
    │   └── datapacks.py                  <- divide scraped data intro train/test/val sets
    ├── tests
    │   ├── test_files                    <- files needed to run tests
    │   └── unit                          <- tests
    ├── sample_detection                  <- source code for this package
    │   ├── __init__.py   
    │   ├── detect                        <- detecting samples
    │   │   ├── __init__.py               
    │   │   ├── audio.py                  <- helper class when loading audio
    │   │   ├── embedding_generators.py   <- generate embeddings from audio
    │   │   ├── sample_detector.py        <- detect samples in two pieces of audio
    │   │   └── sample_loader.py          <- load samples from scraper
    │   ├── scrape                       <- scraping data to detect samples
    │   │   ├── __init__.py               
    │   │   ├── base.py                   <- base scraper class, fetches pages
    │   │   ├── scraper.py                <- Files needed to run the tests
    │   │   ├── whosampled.py             <- scrapes relevant info from whosamples
    │   │   └── youtube.py                <- downloads audio from youtube
    │   ├── cli.py                        <- Scripts to download or generate data    
    │   └── load.py                       <- helps load in scraped data
    ├── .gitignore                        <- files git should ignore
    ├── LICENSE                           <- MIT license
    ├── README.md                         <- top-level README for developers using this project
    ├── requirements-dev.txt              <- requirements needed for dev work
    ├── requirements.txt                  <- requirements needed to use the library  
    ├── setup.cfg                         <- make project pip-installable
    └── setup.py                          <- make project pip-installable

Using the command line interface
------------
To scrape WhoSampled for samples, download the songs from YouTube, and save this all to disk, run 
```
sample-detection scrape <SAVE_DIR> --start-year <START_YEAR> --end-year <END_YEAR> --pages-per-year <PAGES_PER_YEAR>
```
Note: <START_YEAR> and <END_YEAR> refer to the release year of the song that contains the sample. <PAGES_PER_YEAR> dictates the number of pages to scrape on WhoSampled for each year.

To train a model to detect samples, run
```
sample-detection train <INFO_PATH> <AUDIO_DIR> <SAVE_DIR> --sample-rate <SAMPLE_RATE> --sample-duration <SAMPLE_DURATION> --min-negatives <MIN_NEGATIVES>
```
Note: <INFO_PATH> is the path to sample information scraped from WhoSampled by the scraper, <AUDIO_DIR> is the directory containing the audio files scraped by the scraper, and <SAVE_DIR> is the directory in which the resulting model should be saved. <SAMPLE_RATE> is the sample rate at which songs should be loaded (set to 16000 by default), <SAMPLE_DURATION> is the default length of a sample (defaults to 15 seconds), and <MIN_NEGATIVES> is the number of negative samples to generate per positive sample (defaults to 1).

To detect whether one song contains a sample of another, run
```
sample-detection detect <SAMPLE_DETECTOR_PATH> <AUDIO_1_PATH> <AUDIO_2_PATH> --hop-length <HOP_LENGTH> --threshold <THRESHOLD>
```
Note: <SAMPLE_DETECTOR_PATH> is the path to a pre-trained sample detector, as generated by `sample-detection train`. <AUDIO_1_PATH> and <AUDIO_2_PATH> are paths to songs/audio files. <HOP_LENGTH> is the amount of time (in seconds) between the start of each potential sample (defaults to 3), and <THRESHOLD> is the value `x` such that if the score from the sample detector is greater than `x`, we consider there to be a sample (defaults to 1 - 1e-5). 

Using the python package
------------

- Using the scraper

Scraping samples is easy, and can be done as below. If you don't want to scrape WhoSampled and YouTube at the same time, or only want to scrape one of these, there are scrapers to handle these as well - more information can be found in the docs.

```
from sample_detection.scrape.scraper import SampleScraper

scraper = SampleScraper(save_dir=save_dir)
scraper.scrape(
    start_year=start_year, end_year=end_year, pages_per_year=pages_per_year
)
```

- Building a sample detection model

Once you've scraped the data you need, building and using a sample detection model is also easy:

```
from sample_detection.detect.audio import Audio
from sample_detection.detect.sample_detector import SampleDetector
from sample_detection.scrape.load import load_sample_info

train_df = load_sample_info(scraped_info_path)

model = SampleDetector(sample_duration=sample_duration, sample_rate=sample_rate)
model.fit(sample_info=train_df, audio_dir=scraped_audio_dir, min_negatives=min_negatives)

audio_1 = Audio(path=audio_1_path)
audio_2 = Audio(path=audio_2_path)

samples = model.find_samples(
    audio_1=audio_1, audio_2=audio_2, threshold=threshold, hop_length=hop_length
)
```

- Support for custom embeddings

While Sample Detection uses Wav2CLIP embeddings, it is possible to use embeddings from other models by creating a custom embedding generator. All you need to do is create a subclass of EmbeddingGenerator, and implement the `generate_embedding` function, which takes in audio as a 1D np.ndarray, and returns an embedding. You can then pass this to your sample detector and train your sample detector as normal. An outline of what this looks like is below, but more detail can be found in the docs.

```
from sample_detection.detect.sample_detector import SampleDetector
from sample_detection.scrape.load import load_sample_info

class CustomEmbeddingGenerator(EmbeddingGenerator):

    def generate_embedding(self, audio_array: np.ndarray) -> np.ndarray:

        embedding = np.array([0 for i in range(512)])

        return embedding

custom_embedding_generator = CustomEmbeddingGenerator()

model = SampleDetector(
    sample_duration=15,
    sample_rate=16000,
    embedding_generator=custom_embedding_generator,
)

train_df = load_sample_info(scraped_info_path)
model.fit(sample_info=train_df, audio_dir=scraped_audio_dir, min_negatives=min_negatives)
```