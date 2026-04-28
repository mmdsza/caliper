# EasyTrain

A grader/eval authoring stack for LLM post-training. Author Python graders in a Monaco-based browser editor, dry-run them against versioned eval sets, mutation-test for reward-hack holes, compute Cohen's κ vs reference labels, and compile to OpenAI's reinforcement-fine-tuning JSON spec.

## Layout

```
.
├── packages/
│   ├── sdk/          @grader decorator, Runner, typed records, source loader   (Python)
│   ├── store/        versioned content-addressed eval-set storage (SQLite/PG)  (Python)
│   ├── rft_compile/  spec → OpenAI RFT JSON, RFTRunner, OpenAI client          (Python)
│   ├── differ/       v1 vs v2 grader diff on an eval set                       (Python)
│   ├── monitor/      reward-hack auxiliary judges (4 built-in rubrics)         (Python)
│   ├── mutate/       mutation testing for graders (8 built-ins)                (Python)
│   ├── kappa/        Cohen's κ between a grader and a labeled reference set   (Python)
│   ├── server/       FastAPI dry-run backend + pluggable executor (in-proc/E2B)(Python)
│   └── editor/       Monaco-based grader IDE                                   (Next.js)
└── scripts/
    ├── dev.sh                      starts server (:8000) + editor (:3456) together
    ├── smoke_dry_run.py            POSTs sample grader → /dry-run, asserts shape
    ├── smoke_mutation_test.py      runs all 8 mutations against robust + naked-gold graders
    ├── smoke_rft_passthrough.py    compile → submit → poll via FakeRFTClient
    ├── smoke_kappa_tracking.py     legal grader vs hand-built labels; prints unweighted/linear/quadratic κ
    └── smoke_e2b_sandbox.py        live smoke for E2BExecutor; skips without E2B_API_KEY
```

## Quickstart

**Run both servers (editor + dry-run backend):**

```bash
./scripts/dev.sh
# editor: http://localhost:3456
# server: http://localhost:8000
```

Open the editor, edit the sample grader, click **Dry run** — see the histogram, top/bottom rows, and any failures.

**Or in pieces:**

```bash
# Python workspace — uv ≥ 0.11
uv sync
uv run pytest -q              # 258 tests
uv run ruff check packages

# Editor — pnpm
pnpm --dir packages/editor install
pnpm --dir packages/editor dev
```

## Smokes

Self-contained end-to-end checks. No network, no API keys (one optional smoke skips cleanly without `E2B_API_KEY`).

```bash
uv run python scripts/smoke_dry_run.py
uv run python scripts/smoke_mutation_test.py
uv run python scripts/smoke_rft_passthrough.py
uv run python scripts/smoke_kappa_tracking.py
uv run python scripts/smoke_e2b_sandbox.py    # SKIPs without E2B_API_KEY
```

## What's interesting in here

- **Mutation testing for graders** (`packages/mutate/`) — 8 built-in mutations (drop-format-gate, wrong-answer, truncate, whitespace-perturb, prepend-sycophancy, append-garbage, case-perturb, hardcode-gold) plus a rank-order CI gate. Surfaces graders that pass on the surface but accept reward-hacked outputs.
- **Cohen's κ tracking** (`packages/kappa/`) — unweighted / linear-weighted / quadratic-weighted κ between a grader and a labeled reference set, with explicit `kappa_undefined` flag for the math-undefined cases. Pure stdlib + Pydantic.
- **Pluggable grader executor** (`packages/server/src/easytrain_server/executor.py`) — `InProcessExecutor` for offline tests, `E2BExecutor` runs grader source inside a Firecracker microVM with the workspace SDK shipped in as a wheel. Selected via `EASYTRAIN_GRADER_EXECUTOR=in_process|e2b`.
- **Versioned content-addressed eval-set storage** (`packages/store/`) — `EvalSetStore` Protocol, two backends (`InMemoryStore` + `SqlStore` via SQLAlchemy 2.x over SQLite or Postgres), 12-char SHA-256 content hashes, idempotent `create()`. Selected via `EASYTRAIN_STORE_URL`.
- **OpenAI RFT passthrough** (`packages/rft_compile/`) — multi-grader specs compile to OpenAI's `method.type=reinforcement` wire format; `RFTRunner` + `OpenAIRFTClient` (mocked SDK tests, no API key needed) submit and poll through `queued → running → succeeded`.

## Stack

- Python ≥ 3.12, uv workspace, ruff, pytest with `--import-mode=importlib`
- TypeScript / pnpm / Next.js 15 (app router) / React 19 / `@monaco-editor/react`
- Pydantic v2 for all typed records
- SQLAlchemy 2.x for the eval-set store; `e2b-code-interpreter` as an optional `[e2b]` extra

## License

[Apache-2.0](./LICENSE).
