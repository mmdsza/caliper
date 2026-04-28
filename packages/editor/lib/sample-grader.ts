export interface SampleGrader {
  name: string;
  version: string;
  source: string;
}

export const sampleGrader: SampleGrader = {
  name: "legal_citation_grader",
  version: "0.1.0",
  source: `from caliper_sdk import grader, Rollout, GraderResult


@grader(name="legal_citation_grader", version="0.1.0")
def legal_citation_grader(rollout: Rollout) -> GraderResult:
    response = rollout.response
    if not response.startswith("<think>"):
        return GraderResult(score=0.0, explanation="missing <think> gate")

    # ... extract answer, compare to gold
    answer = response.split("</think>")[-1].strip()
    match = rollout.gold is not None and answer == rollout.gold
    return GraderResult(
        score=1.0 if match else 0.0,
        explanation="match" if match else f"expected {rollout.gold!r}, got {answer!r}",
    )
`,
};
