"""EasyTrain RFT — typed grader specs, JSON compiler, and OpenAI passthrough.

See ``docs/03-graders-and-rewards.md`` for the format reference.

Public surface:
    Spec types          — StringCheckGrader, ScoreModelGrader, MultiGrader, ...
    Compiler            — compile_to_rft, compile_to_rft_json, validate_calculate_output
    Runner              — RFTRunner, build_payload, Job, JobStatus, RFTClient
    Real client         — OpenAIRFTClient (lazy openai import)
    Test double         — FakeRFTClient (in easytrain_rft.testing)
"""

from easytrain_rft.compile import (
    compile_to_rft,
    compile_to_rft_json,
    validate_calculate_output,
)
from easytrain_rft.runner import (
    Job,
    JobStatus,
    RFTClient,
    RFTRunner,
    build_payload,
)
from easytrain_rft.spec import (
    AnyGrader,
    LabelModelGrader,
    MultiGrader,
    PythonGrader,
    ScoreModelGrader,
    StringCheckGrader,
    TextSimilarityGrader,
)

__all__ = [
    "AnyGrader",
    "Job",
    "JobStatus",
    "LabelModelGrader",
    "MultiGrader",
    "PythonGrader",
    "RFTClient",
    "RFTRunner",
    "ScoreModelGrader",
    "StringCheckGrader",
    "TextSimilarityGrader",
    "build_payload",
    "compile_to_rft",
    "compile_to_rft_json",
    "validate_calculate_output",
]
