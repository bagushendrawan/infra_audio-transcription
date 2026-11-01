import json
import logging
import re
import subprocess
import sys
from pathlib import Path

from src.transcription.config import Settings
from src.transcription.schemas import make_result


def infer(audio, language, settings):
    import av
    import faster_whisper
    import torch
    from ctc_forced_aligner import (
        generate_emissions,
        get_alignments,
        get_spans,
        load_alignment_model,
        postprocess_results,
        preprocess_text,
    )
    from deepmultilingualpunctuation import PunctuationModel

    from src.transcription.vendor.helpers import (
        get_realigned_ws_mapping_with_punctuation,
        get_sentences_speaker_mapping,
        get_words_speaker_mapping,
        langs_to_iso,
        punct_model_langs,
    )
    from src.transcription.vendor.msdd import MSDDDiarizer

    try:
        with av.open(str(audio)) as container:
            if not container.streams.audio:
                return {"status": 400, "error": "No audio stream"}
            stream = container.streams.audio[0]
            if stream.duration and stream.duration * stream.time_base > settings.max_duration:
                return {"status": 413, "error": "Audio exceeds duration limit"}
        # Decode in bounded chunks before materializing the waveform.
        with av.open(str(audio)) as container:
            duration = 0.0
            for frame in container.decode(audio=0):
                duration += frame.samples / frame.sample_rate
                if duration > settings.max_duration:
                    return {"status": 413, "error": "Audio exceeds duration limit"}
        if not duration:
            return {"status": 400, "error": "Empty audio"}
    except (av.error.FFmpegError, ValueError):
        return {"status": 400, "error": "Invalid audio"}
    if not torch.cuda.is_available():
        return {"status": 503, "error": "CUDA unavailable"}
    if settings.model.endswith(".en") and language not in (None, "en"):
        return {"status": 422, "error": "Configured model supports English only"}
    vocal_target = str(audio)
    if settings.separation:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "demucs.separate",
                "-n",
                "htdemucs",
                "--two-stems=vocals",
                str(audio),
                "-o",
                str(audio.parent),
                "--device",
                "cuda",
            ],
            check=True,
        )
        vocal_target = str(audio.parent / "htdemucs" / audio.stem / "vocals.wav")

    whisper_model = faster_whisper.WhisperModel(
        settings.model, device="cuda", compute_type="float16"
    )
    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
    audio_waveform = faster_whisper.decode_audio(vocal_target)
    transcript_segments, info = whisper_pipeline.transcribe(
        audio_waveform,
        language=language,
        suppress_tokens=[-1],
        batch_size=settings.batch_size,
    )

    full_transcript = "".join(segment.text for segment in transcript_segments)
    if not full_transcript.strip():
        return make_result(None, duration, [])

    # clear gpu vram
    del whisper_model, whisper_pipeline
    torch.cuda.empty_cache()

    # Forced Alignment
    alignment_model, alignment_tokenizer = load_alignment_model(
        "cuda",
        dtype=torch.float16,
    )

    emissions, stride = generate_emissions(
        alignment_model,
        torch.from_numpy(audio_waveform).to(alignment_model.dtype).to(alignment_model.device),
        batch_size=settings.batch_size,
    )

    del alignment_model
    torch.cuda.empty_cache()

    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language=langs_to_iso[info.language],
    )

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_token)

    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    diarizer_model = MSDDDiarizer(device="cuda")

    speaker_ts = diarizer_model.diarize(torch.from_numpy(audio_waveform).unsqueeze(0))
    del diarizer_model
    torch.cuda.empty_cache()

    if not speaker_ts:
        raise RuntimeError("Diarization produced no speakers for transcribed speech")
    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if info.language in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        punct_model = PunctuationModel(model="kredor/punctuate-all")

        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        def is_acronym(word):
            return re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", word)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

    else:
        logging.warning(
            f"Punctuation restoration is not available for {info.language} language."
            " Using the original punctuation."
        )

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    return make_result(info.language, duration, ssm)


if __name__ == "__main__":
    audio, output, language = sys.argv[1:]
    result = infer(Path(audio), language or None, Settings())
    Path(output).write_text(json.dumps(result), encoding="utf-8")
