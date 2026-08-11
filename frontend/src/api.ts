export type Citation = {
  label: string;
  target_chunk_id: string;
  ticker: string;
  record_type: string;
  report_date: string;
  filing_date: string;
  section_title: string;
  page_start: string | number | null;
  page_end: string | number | null;
  display_page_start: string | number | null;
  display_page_end: string | number | null;
  source_url: string;
};

export type EvidenceRecord = {
  target_chunk_id: string;
  ticker?: string;
  record_type?: string;
  document?: string;
  evidence?: string;
  retrieval_text?: string;
  metadata?: Record<string, unknown>;
};

export type AnswerResponse = {
  question: string;
  ticker: string | null;
  status: "supported" | "ambiguous" | "unsupported";
  answer_type: "numeric" | "narrative";
  answer: string;
  reason: string;
  citations: Citation[];
  evidence: EvidenceRecord[];
};

type ErrorResponse = { error?: string };

export async function requestAnswer(
  question: string,
  sessionTicker: string | null,
  signal?: AbortSignal,
): Promise<AnswerResponse> {
  let response: Response;
  try {
    response = await fetch("/api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_ticker: sessionTicker }),
      signal,
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    const serviceError = new Error(
      "The answer service is not running. Start it with `npm run api` in the frontend folder.",
    ) as Error & { cause: unknown };
    serviceError.cause = error;
    throw serviceError;
  }

  const payload = await response.json().catch(() => ({})) as AnswerResponse | ErrorResponse;
  if (!response.ok) {
    throw new Error("error" in payload && payload.error ? payload.error : "The answer service returned an error.");
  }
  return payload as AnswerResponse;
}
