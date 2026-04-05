"""RunPod predictor implementation backed by WhisperX."""

import gc
import os
import threading
from typing import Any, Optional

import numpy as np
import whisperx
from runpod.serverless.utils import rp_cuda

AVAILABLE_MODELS = {
    "tiny",
    "base",
    "small",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "distil-large-v2",
    "distil-large-v3",
    "turbo",
}

DEFAULT_COMPUTE_TYPE = "float16"


class Predictor:
    """A Predictor class for WhisperX with lazy model loading."""

    def __init__(self):
        self.asr_models: dict[tuple[str, Optional[str], str], Any] = {}
        self.align_models: dict[str, tuple[Any, Any]] = {}
        self.diarize_model: Any = None
        self.model_lock = threading.Lock()

    def setup(self):
        """No models are pre-loaded. Setup is minimal."""
        pass

    def _device(self) -> str:
        return "cuda" if rp_cuda.is_available() else "cpu"

    def _clear_model_cache(self) -> None:
        self.asr_models.clear()
        self.align_models.clear()
        self.diarize_model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _get_asr_model(
        self,
        model_name: str,
        language: Optional[str],
        task: str,
        asr_options: dict[str, Any],
        enable_vad: bool,
    ) -> Any:
        cache_key = (
            model_name,
            language,
            task,
            enable_vad,
            _freeze(asr_options),
        )

        with self.model_lock:
            if cache_key not in self.asr_models:
                if self.asr_models:
                    existing_models = ", ".join(
                        f"{name}:{lang or 'auto'}:{model_task}"
                        for name, lang, model_task in self.asr_models
                    )
                    print(f"Unloading models: {existing_models}...")
                    self._clear_model_cache()
                    print("Model cache cleared.")

                print(f"Loading model: {model_name} ({task})...")
                self.asr_models[cache_key] = whisperx.load_model(
                    model_name,
                    self._device(),
                    compute_type=DEFAULT_COMPUTE_TYPE if self._device() == "cuda" else "int8",
                    language=language,
                    task=task,
                    asr_options=asr_options,
                    vad_method="silero" if enable_vad else "silero",
                )
                print(f"Model {model_name} ({task}) loaded successfully.")

            return self.asr_models[cache_key]

    def _get_align_model(self, language_code: str) -> tuple[Any, Any]:
        with self.model_lock:
            if language_code not in self.align_models:
                self.align_models[language_code] = whisperx.load_align_model(
                    language_code=language_code,
                    device=self._device(),
                )
            return self.align_models[language_code]

    def _get_diarize_model(self) -> Any:
        with self.model_lock:
            if self.diarize_model is None:
                from whisperx.diarize import DiarizationPipeline

                self.diarize_model = DiarizationPipeline(
                    token=_get_hf_token(),
                    device=self._device(),
                )
            return self.diarize_model

    def predict(
        self,
        audio,
        model_name="base",
        transcription="plain_text",
        translate=False,
        translation="plain_text",
        language=None,
        temperature=0,
        best_of=5,
        beam_size=5,
        patience=1,
        length_penalty=None,
        suppress_tokens="-1",
        initial_prompt=None,
        condition_on_previous_text=True,
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        enable_vad=True,
        word_timestamps=False,
        speaker_id=False,
        batch_size=8,
    ):
        """Run a single prediction using WhisperX."""
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model name: {model_name}. Available models are: {AVAILABLE_MODELS}"
            )

        if temperature_increment_on_fallback is not None:
            temperatures = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperatures = [temperature]

        audio_data = whisperx.load_audio(str(audio))
        asr_options = {
            "beam_size": beam_size,
            "best_of": best_of,
            "patience": patience,
            "length_penalty": length_penalty,
            "temperatures": temperatures,
            "compression_ratio_threshold": compression_ratio_threshold,
            "log_prob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "condition_on_previous_text": condition_on_previous_text,
            "initial_prompt": initial_prompt,
            "suppress_tokens": [int(token) for token in suppress_tokens.split(",") if token],
        }

        transcription_model = self._get_asr_model(
            model_name,
            language,
            "transcribe",
            asr_options,
            enable_vad,
        )
        transcription_result = transcription_model.transcribe(
            audio_data,
            batch_size=batch_size,
            print_progress=False,
        )

        aligned_result = None
        if (word_timestamps or speaker_id) and transcription_result.get("segments"):
            align_language = transcription_result.get("language") or language or "en"
            align_model, align_metadata = self._get_align_model(align_language)
            aligned_result = whisperx.align(
                transcription_result["segments"],
                align_model,
                align_metadata,
                audio_data,
                self._device(),
                return_char_alignments=False,
            )

        if speaker_id:
            diarize_model = self._get_diarize_model()
            diarize_segments = diarize_model(str(audio))
            base_result = aligned_result or transcription_result
            transcription_result = whisperx.assign_word_speakers(diarize_segments, base_result)
        elif aligned_result is not None:
            transcription_result = aligned_result

        transcription_segments = serialize_segments(transcription_result.get("segments", []))
        transcription_output = format_segments(transcription, transcription_segments)

        translation_output = None
        if translate:
            translation_model = self._get_asr_model(
                model_name,
                language,
                "translate",
                asr_options,
                enable_vad,
            )
            translation_result = translation_model.transcribe(
                audio_data,
                batch_size=batch_size,
                print_progress=False,
            )
            translation_segments = serialize_segments(translation_result.get("segments", []))
            translation_output = format_segments(translation, translation_segments)

        results = {
            "segments": _drop_words(transcription_segments),
            "detected_language": transcription_result.get("language") or language,
            "transcription": transcription_output,
            "translation": translation_output,
            "device": self._device(),
            "model": model_name,
        }

        if word_timestamps:
            results["word_timestamps"] = collect_word_timestamps(transcription_result.get("segments", []))

        return results


def _get_hf_token() -> Optional[str]:
    return (
        os.getenv("hf_token")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(val)) for key, val in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _drop_words(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for segment in segments:
        segment.pop("words", None)
    return segments


def collect_word_timestamps(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timestamps = []
    for segment in segments:
        for word in segment.get("words", []):
            timestamps.append(
                {
                    "word": word.get("word"),
                    "start": word.get("start"),
                    "end": word.get("end"),
                    **({"speaker": word.get("speaker")} if word.get("speaker") is not None else {}),
                }
            )
    return timestamps


def serialize_segments(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize segments for the API response."""
    serialized = []
    for index, segment in enumerate(transcript):
        serialized_segment = {
            "id": segment.get("id", index),
            "start": segment.get("start"),
            "end": segment.get("end"),
            "text": segment.get("text", ""),
        }
        if segment.get("speaker") is not None:
            serialized_segment["speaker"] = segment["speaker"]
        if segment.get("words"):
            serialized_segment["words"] = segment["words"]
        serialized.append(serialized_segment)
    return serialized


def format_segments(format_type: str, segments: list[dict[str, Any]]) -> str:
    """Format segments to the desired output type."""
    format_type = format_type.strip().replace(" ", "_")
    if format_type == "plain_text":
        return " ".join(segment.get("text", "").lstrip() for segment in segments).strip()
    if format_type == "formatted_text":
        return "\n".join(segment.get("text", "").lstrip() for segment in segments).strip()
    if format_type == "srt":
        return write_srt(segments)
    if format_type == "vtt":
        return write_vtt(segments)

    print(f"Warning: Unknown format '{format_type}', defaulting to plain text.")
    return " ".join(segment.get("text", "").lstrip() for segment in segments).strip()


def format_timestamp(
    seconds: Optional[float],
    always_include_hours: bool = False,
    decimal_marker: str = ".",
) -> str:
    if seconds is None:
        seconds = 0.0

    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    secs = milliseconds // 1000
    milliseconds -= secs * 1000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{secs:02d}{decimal_marker}{milliseconds:03d}"


def write_vtt(transcript: list[dict[str, Any]]) -> str:
    result = "WEBVTT\n\n"

    for segment in transcript:
        result += (
            f"{format_timestamp(segment.get('start'), always_include_hours=True)} --> "
            f"{format_timestamp(segment.get('end'), always_include_hours=True)}\n"
        )
        result += f"{segment.get('text', '').strip().replace('-->', '->')}\n\n"

    return result


def write_srt(transcript: list[dict[str, Any]]) -> str:
    result = ""

    for index, segment in enumerate(transcript, start=1):
        result += f"{index}\n"
        result += (
            f"{format_timestamp(segment.get('start'), always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment.get('end'), always_include_hours=True, decimal_marker=',')}\n"
        )
        result += f"{segment.get('text', '').strip().replace('-->', '->')}\n\n"

    return result
