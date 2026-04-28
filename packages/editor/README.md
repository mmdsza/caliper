# @easytrain/editor

Monaco-based grader editor for EasyTrain — the authoring surface for `@grader`
reward functions. Sprint 0 ships a pure client-side scaffold; backend wiring,
SDK type-checking, and dry-run execution land in later sprints.

## Run

```bash
pnpm install
pnpm dev
# open http://localhost:3000
```

Other scripts:

- `pnpm typecheck` — `tsc --noEmit`
- `pnpm build` — production Next.js build
- `pnpm start` — serve the production build

## What's wired up

- Next.js 15 (app router) + React 19, TypeScript strict mode.
- `@monaco-editor/react` mounted in a client component (`components/GraderEditor.tsx`),
  language `python`, theme `vs-dark`, `automaticLayout: true`, no minimap.
- A sample `@grader` (`lib/sample-grader.ts`) — the "format gate + correctness"
  pattern from `docs/03-graders-and-rewards.md`, adapted to the EasyTrain SDK
  shape (`from easytrain_sdk import grader, Rollout, GraderResult`).
- Drafts persist to `localStorage` under the key `easytrain.editor.draft`; on
  reload, the editor hydrates from storage and falls back to the sample grader
  when no draft exists.

## What's not wired up yet

- No backend — nothing is sent off-machine; there's no save endpoint, version
  history, or grader registry yet.
- No live SDK type-checking — the editor doesn't know about `easytrain_sdk`'s
  symbols. Syntax/type errors are not surfaced.
- No dry-run / "Run on sample rollouts" button. That requires the
  `rft_compile` + sandbox stack.
- No multi-file projects, no diffing against a previous version, no monitor
  hooks. Those land alongside `packages/differ/` and `packages/monitor/`.
