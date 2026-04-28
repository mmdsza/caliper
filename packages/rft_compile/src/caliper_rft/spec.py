"""Pydantic spec models for OpenAI RFT graders.

One model per native grader type, plus a discriminated union ``AnyGrader``.
All models forbid extra fields so schema drift fails loudly.

Reference: docs/03-graders-and-rewards.md, section
"OpenAI's RFT JSON spec — the de facto interchange format".
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StringCheckGrader(BaseModel):
    """Programmatic string equality / prefix / suffix / substring check."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["string_check"] = "string_check"
    operation: Literal["eq", "ne", "starts_with", "ends_with", "contains"]
    reference: str


class TextSimilarityGrader(BaseModel):
    """Reference-based similarity (BLEU/ROUGE/fuzzy/embedding cosine)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text_similarity"] = "text_similarity"
    evaluation_metric: Literal[
        "fuzzy_match",
        "bleu",
        "rouge_1",
        "rouge_2",
        "rouge_3",
        "rouge_4",
        "rouge_5",
        "rouge_l",
        "meteor",
        "cosine",
    ]
    reference: str
    pass_threshold: float | None = None


class ScoreModelGrader(BaseModel):
    """LLM-as-judge that emits a numeric score within ``range``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["score_model"] = "score_model"
    model: str
    prompt: str
    range: tuple[float, float] = (0.0, 1.0)


class LabelModelGrader(BaseModel):
    """LLM-as-judge that emits a label from a closed set."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["label_model"] = "label_model"
    model: str
    prompt: str
    labels: list[str]
    passing_labels: list[str]


class PythonGrader(BaseModel):
    """Inline Python source defining ``grade(sample, item) -> float``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["python"] = "python"
    source: str


class MultiGrader(BaseModel):
    """Composite grader: arithmetic over named sub-graders."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["multi"] = "multi"
    graders: dict[str, AnyGrader]
    calculate_output: str

    @model_validator(mode="after")
    def _check_calculate_output(self) -> MultiGrader:
        # Local import to avoid a circular import at module load time.
        from caliper_rft.compile import validate_calculate_output

        validate_calculate_output(self.calculate_output, set(self.graders.keys()))
        return self


AnyGrader = Annotated[
    StringCheckGrader
    | TextSimilarityGrader
    | ScoreModelGrader
    | LabelModelGrader
    | PythonGrader
    | MultiGrader,
    Field(discriminator="type"),
]


# Resolve the forward reference inside MultiGrader.graders.
MultiGrader.model_rebuild()
