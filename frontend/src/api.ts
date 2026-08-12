export type Citation = {
  citation_id: string;
  label: string;
  target_chunk_id: string;
  ticker: string;
  record_type: string;
  report_date?: string;
  filing_date?: string;
  section_title?: string;
  page_start?: string | number | null;
  page_end?: string | number | null;
  display_page_start?: string | number | null;
  display_page_end?: string | number | null;
  source_url?: string;
};

export type AnswerResponse = {
  question: string;
  ticker: string | null;
  status: "supported" | "ambiguous" | "unsupported";
  answer_type: "numeric" | "narrative";
  answer: string;
  reason: string;
  citations: Citation[];
};

export type ThreadSummary = {
  id: string;
  title: string;
  session_ticker: string | null;
  created_at: string;
  updated_at: string;
};

export type Turn = {
  id: string;
  question: string;
  state: "loading" | "answered" | "error";
  response?: AnswerResponse;
  error?: string;
  error_code?: string;
  status?: string;
  created_at?: string;
};

export type SourceChunk = {
  target_chunk_id: string;
  role: "previous" | "anchor" | "next";
  record_type: string;
  document: string;
  metadata: Record<string, unknown>;
};

export type CitationContext = {
  citation: { id: string; label: string; metadata: Citation };
  target_chunk_id: string;
  record_type: string;
  ticker: string;
  source_url: string;
  corpus_hash: string;
  chunks: SourceChunk[];
};

type ThreadHistory = {
  thread: ThreadSummary;
  turns: Turn[];
};

type ErrorPayload = { error?: string; detail?: string; code?: string };

export class ApiError extends Error {
  code?: string;
  status: number;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (error) {
    const serviceError = new ApiError(
      "The answer service is not running. Start it with `npm run api` in the frontend folder.",
      0,
    ) as ApiError & { cause: unknown };
    serviceError.cause = error;
    throw serviceError;
  }
  const payload = await response.json().catch(() => ({})) as T & ErrorPayload;
  if (!response.ok) {
    throw new ApiError(
      payload.error || payload.detail || "The answer service returned an error.",
      response.status,
      payload.code,
    );
  }
  return payload;
}

export async function listThreads(signal?: AbortSignal): Promise<ThreadSummary[]> {
  const payload = await requestJson<{ threads: ThreadSummary[] }>("/api/threads", { signal });
  return payload.threads;
}

export function createThread(): Promise<ThreadSummary> {
  return requestJson("/api/threads", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
}

export function loadThread(threadId: string, signal?: AbortSignal): Promise<ThreadHistory> {
  return requestJson(`/api/threads/${threadId}/messages`, { signal });
}

export function renameThread(threadId: string, title: string): Promise<ThreadSummary> {
  return requestJson(`/api/threads/${threadId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
}

export async function deleteThread(threadId: string): Promise<void> {
  const response = await fetch(`/api/threads/${threadId}`, { method: "DELETE" });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({})) as ErrorPayload;
    throw new ApiError(payload.detail || "Conversation could not be deleted.", response.status);
  }
}

type StreamEvent =
  | { type: "status"; stage: string; message: string }
  | { type: "answer"; turn: Turn }
  | { type: "error"; turn: Turn; error: string; code: string }
  | { type: "done"; status: number };

export async function streamAnswer(
  threadId: string,
  question: string,
  onStatus: (stage: string, message: string) => void,
  signal?: AbortSignal,
): Promise<Turn> {
  const response = await fetch(`/api/threads/${threadId}/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!response.ok || !response.body) {
    const payload = await response.json().catch(() => ({})) as ErrorPayload;
    throw new ApiError(
      payload.detail || payload.error || "The answer stream could not start.",
      response.status,
      payload.code,
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalTurn: Turn | undefined;

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      const data = block.split("\n").find((line) => line.startsWith("data: "))?.slice(6);
      if (!data) continue;
      const event = JSON.parse(data) as StreamEvent;
      if (event.type === "status") onStatus(event.stage, event.message);
      if (event.type === "answer" || event.type === "error") finalTurn = event.turn;
    }
    if (done) break;
  }
  if (!finalTurn) throw new ApiError("The answer stream ended without a result.", 500);
  return finalTurn;
}

export function loadCitationContext(
  citationId: string,
  signal?: AbortSignal,
): Promise<CitationContext> {
  return requestJson(`/api/citations/${citationId}/context?radius=1`, { signal });
}
