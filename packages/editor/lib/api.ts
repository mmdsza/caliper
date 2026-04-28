// Thin client for the dry-run backend. Calls go through Next.js rewrites
// (/api/* → http://127.0.0.1:8000/*) so this module never needs to know
// the server URL directly.

export interface RowResult {
  row_id: string;
  score: number;
  explanation: string | null;
  components: Record<string, number>;
}

export interface RowFailure {
  row_id: string;
  error_type: string;
  error_message: string;
}

export interface RunReport {
  grader_name: string;
  grader_version: string;
  n_rows: number;
  n_succeeded: number;
  n_failed: number;
  mean_score: number;
  std_score: number;
  min_score: number;
  max_score: number;
  results: RowResult[];
  failures: RowFailure[];
}

export interface EvalRowOut {
  id: string;
  prompt: string;
  response: string;
  gold: string | null;
}

export interface DryRunResponse {
  grader_name: string;
  grader_version: string;
  eval_set: string;
  report: RunReport;
  rows: EvalRowOut[];
}

export interface DryRunError {
  status: number;
  detail: string;
}

export async function dryRun(
  source: string,
  evalSet: string,
): Promise<DryRunResponse> {
  const r = await fetch("/api/dry-run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source, eval_set: evalSet }),
  });
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try {
      const body = (await r.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // ignore
    }
    const err: DryRunError = { status: r.status, detail };
    throw err;
  }
  return (await r.json()) as DryRunResponse;
}
