import logging
import pandas as pd

from ast import literal_eval

from sample_detection.detect.embedding_generators.wav2clip.generator import (
    Wav2ClipEmbeddingGenerator,
)

logging.root.setLevel(logging.INFO)


def test_generate_embeddings_from_directory(
    embedding_generator, audio_dir, sample_details_path
):
    sample_info = pd.read_csv(sample_details_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    sample_info, embeddings = embedding_generator.generate_embeddings_from_directory(
        sample_info=sample_info, audio_dir=str(audio_dir)
    )

    sample_file_names = set(sample_info["sample_from_ytid"]) | set(
        sample_info["sample_in_ytid"]
    )

    df = pd.concat(
        [
            sample_info[["sample_in_ytid", "sample_in_times"]].rename(
                mapper={"sample_in_ytid": "ytid", "sample_in_times": "times"}, axis=1
            ),
            sample_info[["sample_from_ytid", "sample_from_times"]].rename(
                mapper={"sample_from_ytid": "ytid", "sample_from_times": "times"},
                axis=1,
            ),
        ]
    )

    assert all(
        [file_name in sample_file_names for file_name in embeddings.keys()]
    ) and all(
        [(time in embeddings[row["ytid"]])]
        for _, row in df.iterrows()
        for time in row["times"]
    )


def test_generate_embeddings_from_directory_logging(
    caplog, embedding_generator, sample_details_path, audio_dir
):
    sample_info = pd.read_csv(sample_details_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    sample_info, embeddings = embedding_generator.generate_embeddings_from_directory(
        sample_info=sample_info, audio_dir=str(audio_dir)
    )

    sample_file_names = set(sample_info["sample_from_ytid"]) | set(
        sample_info["sample_in_ytid"]
    )

    df = pd.concat(
        [
            sample_info[["sample_in_ytid", "sample_in_times"]].rename(
                mapper={"sample_in_ytid": "ytid", "sample_in_times": "times"}, axis=1
            ),
            sample_info[["sample_from_ytid", "sample_from_times"]].rename(
                mapper={"sample_from_ytid": "ytid", "sample_from_times": "times"},
                axis=1,
            ),
        ]
    )

    assert "Generating embeddings" in caplog.text


def test_wav2clip_embedding_generator_init(clip_length, sample_rate):
    embedding_generator = Wav2ClipEmbeddingGenerator(
        sample_duration=clip_length, sample_rate=sample_rate
    )

    assert True


def test_wav2clip_generate_embedding(clip_length, sample_rate, audio):
    embedding_generator = Wav2ClipEmbeddingGenerator(
        sample_duration=clip_length, sample_rate=sample_rate
    )

    embedding = embedding_generator.generate_embedding(audio)

    assert embedding.shape == (512,)


def test_load_audio(embedding_generator, white_noise_path):
    loaded_audio = embedding_generator._load_audio(
        audio_path=white_noise_path, start_time=0
    )

    assert loaded_audio.ndim == 1


def test_load_audio_logging(caplog, embedding_generator, white_noise_path):
    with caplog.at_level(logging.INFO):
        loaded_audio = embedding_generator._load_audio(
            audio_path=white_noise_path, start_time=0
        )

    assert ("Loading audio from file" in caplog.text) and (
        "Successfully loaded audio from file" in caplog.text
    )


def test_load_sample(embedding_generator, sample_details_path, audio_dir):
    sample_info = pd.read_csv(sample_details_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    sample = sample_info.iloc[0, :]
    sample_loaded_successfully, loaded_sample = embedding_generator.load_sample(
        audio_dir=audio_dir, sample=sample
    )

    sample_in_ytid = sample["sample_in_ytid"]
    sample_from_ytid = sample["sample_from_ytid"]

    assert (
        (sample_in_ytid in loaded_sample)
        and (sample_from_ytid in loaded_sample)
        and all(
            [
                time in loaded_sample[sample_in_ytid]
                for time in sample["sample_in_times"]
            ]
        )
        and all(
            [
                time in loaded_sample[sample_from_ytid]
                for time in sample["sample_from_times"]
            ]
        )
        and sample_loaded_successfully
    )


def test_load_sample_logging(
    caplog, embedding_generator, sample_details_path, audio_dir
):
    sample_info = pd.read_csv(sample_details_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    sample = sample_info.iloc[0, :]
    with caplog.at_level(logging.INFO):
        sample_loaded_successfully, loaded_sample = embedding_generator.load_sample(
            audio_dir=audio_dir, sample=sample
        )

    assert ("Loading sample with whosampled_id" in caplog.text) and (
        "successfully loaded" in caplog.text
    )


def test_load_sample_no_one_to_one_sample_match(
    embedding_generator, sample_details_without_one_to_one_match_path, audio_dir
):
    sample_info = pd.read_csv(sample_details_without_one_to_one_match_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    sample = sample_info.iloc[1, :]
    sample_loaded_successfully, loaded_sample = embedding_generator.load_sample(
        audio_dir=audio_dir, sample=sample
    )

    assert sample_loaded_successfully == False


def test_load_sample_no_one_to_one_sample_match_logging(
    caplog, embedding_generator, sample_details_without_one_to_one_match_path, audio_dir
):
    sample_info = pd.read_csv(sample_details_without_one_to_one_match_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    sample = sample_info.iloc[1, :]
    sample_loaded_successfully, loaded_sample = embedding_generator.load_sample(
        audio_dir=audio_dir, sample=sample
    )

    assert "Could not find a one-to-one match" in caplog.text


def test_load_sample_cannot_load_sample(
    embedding_generator, sample_details_without_one_to_one_match_path, audio_dir
):
    sample_info = pd.read_csv(sample_details_without_one_to_one_match_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    sample = sample_info.iloc[0, :]
    sample_loaded_successfully, loaded_sample = embedding_generator.load_sample(
        audio_dir=audio_dir, sample=sample
    )

    assert sample_loaded_successfully == False


def test_load_sample_cannot_load_sample_logging(
    caplog, embedding_generator, sample_details_without_one_to_one_match_path, audio_dir
):
    sample_info = pd.read_csv(sample_details_without_one_to_one_match_path)
    sample_info["sample_in_times"] = sample_info["sample_in_times"].apply(literal_eval)
    sample_info["sample_from_times"] = sample_info["sample_from_times"].apply(
        literal_eval
    )

    sample = sample_info.iloc[0, :]

    try:
        with caplog.at_level(logging.WARNING):
            sample_loaded_successfully, loaded_sample = embedding_generator.load_sample(
                audio_dir=audio_dir, sample=sample
            )
    except:
        pass

    assert "Could not load sample instance" in caplog.text
