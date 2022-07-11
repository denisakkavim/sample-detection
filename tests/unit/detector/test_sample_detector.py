from pathlib import Path

from sample_detection.model.audio import Audio
from sample_detection.model.model import Model
from sample_detection.detect import SampleDetector

test_files_dir = Path(__file__).resolve().parent.parent.parent / "test_files"
audio_dir = test_files_dir / "audio"


def test_sample_detector(sample_info, clip_length, sample_rate, min_negatives):

    model = Model(sample_duration=clip_length, sample_rate=sample_rate)
    model.fit(sample_info=sample_info, audio_dir=audio_dir, min_negatives=min_negatives)

    audio_1_path = audio_dir / "fan_noise.mp3"
    audio_2_path = audio_dir / "rain_noise.mp3"

    audio_1 = Audio(path=audio_1_path, clip_length=clip_length, sample_rate=sample_rate)
    audio_2 = Audio(path=audio_2_path, clip_length=clip_length, sample_rate=sample_rate)

    sample_detector = SampleDetector(model=model)
    found_samples = sample_detector.find_samples(
        audio_1=audio_1, audio_2=audio_2, threshold=0
    )

    assert all(
        [
            (key in sample)
            for key in ["time_1", "time_2", "confidence"]
            for sample in found_samples
        ]
    )
