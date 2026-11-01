from typing import Literal

from pydantic import BaseModel, Field

Language = Literal[
    "en",
    "zh",
    "de",
    "es",
    "ru",
    "ko",
    "fr",
    "ja",
    "pt",
    "tr",
    "pl",
    "ca",
    "nl",
    "ar",
    "sv",
    "it",
    "id",
    "hi",
    "fi",
    "vi",
    "he",
    "uk",
    "el",
    "ms",
    "cs",
    "ro",
    "da",
    "hu",
    "ta",
    "no",
    "th",
    "ur",
    "hr",
    "bg",
    "lt",
    "la",
    "mi",
    "ml",
    "cy",
    "sk",
    "te",
    "fa",
    "lv",
    "bn",
    "sr",
    "az",
    "sl",
    "kn",
    "et",
    "mk",
    "br",
    "eu",
    "is",
    "hy",
    "ne",
    "mn",
    "bs",
    "kk",
    "sq",
    "sw",
    "gl",
    "mr",
    "pa",
    "si",
    "km",
    "sn",
    "yo",
    "so",
    "af",
    "oc",
    "ka",
    "be",
    "tg",
    "sd",
    "gu",
    "am",
    "yi",
    "lo",
    "uz",
    "fo",
    "ht",
    "ps",
    "tk",
    "nn",
    "mt",
    "sa",
    "lb",
    "my",
    "bo",
    "tl",
    "mg",
    "as",
    "tt",
    "haw",
    "ln",
    "ha",
    "ba",
    "jw",
    "su",
    "yue",
]


class Segment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    speaker: str
    text: str


class Transcription(BaseModel):
    language: str | None
    duration_seconds: float
    text: str
    segments: list[Segment]


def make_result(language, duration, rows):
    speakers = {}
    segments = []
    for row in rows:
        text = row["text"].strip()
        if not text:
            continue
        speaker = speakers.setdefault(row["speaker"], f"SPEAKER_{len(speakers):02d}")
        segments.append(
            dict(
                start=row["start_time"] / 1000,
                end=row["end_time"] / 1000,
                speaker=speaker,
                text=text,
            )
        )
    return dict(
        language=language,
        duration_seconds=duration,
        text=" ".join(s["text"] for s in segments),
        segments=segments,
    )
