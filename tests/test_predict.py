import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FakeModel:
    def __init__(self, task: str):
        self.task = task

    def transcribe(self, audio, batch_size=8, print_progress=False):
        if self.task == "translate":
            return {
                "language": "en",
                "segments": [
                    {"id": 0, "start": 0.0, "end": 1.0, "text": "Translated text"},
                ],
            }
        return {
            "language": "zh",
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.0, "text": "hello"},
            ],
        }


def load_predict_module(monkeypatch):
    fake_whisperx = ModuleType("whisperx")
    fake_whisperx.load_audio = lambda path: "audio-array"
    fake_whisperx.load_model = lambda model_name, device, compute_type, language=None, task="transcribe", **kwargs: FakeModel(task)
    fake_whisperx.load_align_model = lambda language_code, device: ("align-model", {"language": language_code})
    fake_whisperx.align = lambda segments, align_model, align_metadata, audio, device, return_char_alignments=False: {
        "language": align_metadata["language"],
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "hello",
                "words": [{"word": "hello", "start": 0.0, "end": 1.0}],
            }
        ],
    }
    fake_whisperx.assign_word_speakers = lambda diarize_segments, result: {
        "language": result["language"],
        "segments": [
            {
                **result["segments"][0],
                "speaker": "SPEAKER_00",
                "words": [{"word": "hello", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
            }
        ],
    }

    fake_diarize_module = ModuleType("whisperx.diarize")
    fake_numpy = ModuleType("numpy")
    fake_numpy.arange = lambda start, stop, step: [start]

    class FakeDiarizationPipeline:
        called = 0

        def __init__(self, token=None, device="cpu"):
            self.token = token
            self.device = device

        def __call__(self, audio_path):
            FakeDiarizationPipeline.called += 1
            return [{"speaker": "SPEAKER_00"}]

    fake_diarize_module.DiarizationPipeline = FakeDiarizationPipeline

    fake_runpod = ModuleType("runpod")
    fake_serverless = ModuleType("runpod.serverless")
    fake_utils = ModuleType("runpod.serverless.utils")
    fake_utils.rp_cuda = SimpleNamespace(is_available=lambda: False)

    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "whisperx.diarize", fake_diarize_module)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "runpod", fake_runpod)
    monkeypatch.setitem(sys.modules, "runpod.serverless", fake_serverless)
    monkeypatch.setitem(sys.modules, "runpod.serverless.utils", fake_utils)

    sys.modules.pop("predict", None)
    return importlib.import_module("predict"), FakeDiarizationPipeline


def test_predict_skips_diarization_when_speaker_id_false(monkeypatch):
    predict, fake_diarization = load_predict_module(monkeypatch)
    model = predict.Predictor()

    result = model.predict(
        audio="/tmp/audio.wav",
        model_name="base",
        speaker_id=False,
        word_timestamps=False,
    )

    assert fake_diarization.called == 0
    assert result["segments"] == [{"id": 0, "start": 0.0, "end": 1.0, "text": "hello"}]
    assert "word_timestamps" not in result


def test_predict_adds_speaker_labels_when_enabled(monkeypatch):
    predict, fake_diarization = load_predict_module(monkeypatch)
    model = predict.Predictor()

    result = model.predict(
        audio="/tmp/audio.wav",
        model_name="base",
        speaker_id=True,
        word_timestamps=True,
    )

    assert fake_diarization.called == 1
    assert result["segments"] == [
        {
            "id": 0,
            "start": 0.0,
            "end": 1.0,
            "text": "hello",
            "speaker": "SPEAKER_00",
        }
    ]
    assert result["word_timestamps"] == [
        {"word": "hello", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}
    ]


def test_schema_exposes_speaker_id_flag():
    from rp_schema import INPUT_VALIDATIONS

    assert INPUT_VALIDATIONS["speaker_id"]["type"] is bool
    assert INPUT_VALIDATIONS["speaker_id"]["default"] is False
