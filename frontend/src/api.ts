type CitationBase = {
  citation_id: string;
  label: string;
  title?: string;
  source_url?: string;
};

export type FilingCitation = CitationBase & {
  kind: "filing";
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
};

export type WebCitation = CitationBase & {
  kind: "web";
  source_url: string;
  target_chunk_id?: string;
  ticker?: string;
  record_type?: string;
  report_date?: string;
  filing_date?: string;
  section_title?: string;
  page_start?: string | number | null;
  page_end?: string | number | null;
  display_page_start?: string | number | null;
  display_page_end?: string | number | null;
};

export type Citation = FilingCitation | WebCitation;

export type DialogAct =
  | "answer"
  | "clarification"
  | "greeting"
  | "acknowledgement"
  | "capability"
  | "general_explanation"
  | "out_of_scope"
  | "retryable_error"
  | "contextual_transform"
  | "web_answer"
  | "web_research_unavailable"
  | "calculation";

export type NumericFacts = {
  entity: string;
  metric: string;
  variant: string | null;
  period: string;
  value_text: string;
  unit: string;
};

export type BankResult = {
  ticker: string;
  bank_name: string;
  status: "supported" | "ambiguous" | "unsupported";
  answer_type: "numeric" | "narrative";
  answer: string;
  facts: NumericFacts | null;
  reason: string;
  citations: Citation[];
};

export type Diagnostics = {
  route: string;
  agentic_rag_enabled: boolean;
  outcome: string;
  failed_stage?: string | null;
  error_code?: string | null;
  stages: Array<{ stage: string; status: string; latency_ms?: number; [key: string]: unknown }>;
  initial_evidence_count?: number | null;
  final_evidence_count?: number | null;
  model_request_count?: number | null;
  bank_plans: Array<{
    ticker: string;
    action: "agentic_loop" | "generate" | "rewrite_search" | "expand_context" | "abstain";
    final_status?: "sufficient" | "unsupported";
    reason_code?: string;
    explanation?: string;
    rewritten_query?: string | null;
    anchor_target_chunk_id?: string | null;
    model_request_count?: number;
    tool_action_count?: number;
    verifier_request_count?: number;
    fallback?: boolean;
    steps?: Array<{
      action: string;
      query?: string;
      terms?: string[];
      result?: string;
      error_code?: string;
      [key: string]: unknown;
    }>;
  }>;
  quality_gate: { passed: boolean; checks: Record<string, boolean> };
};

export type AnswerResponse = {
  question: string;
  dialog_act?: DialogAct;
  mode?: "comparison";
  ticker: string | null;
  tickers?: string[];
  status: "supported" | "partial" | "ambiguous" | "unsupported";
  answer_type: "numeric" | "narrative";
  answer: string;
  reason: string;
  citations: Citation[];
  bank_results?: BankResult[];
  diagnostics?: Diagnostics;
};

export type ThreadSummary = {
  id: string;
  title: string;
  session_ticker: string | null;
  session_tickers: string[];
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
  diagnostics?: Diagnostics;
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
  citation: { id: string; label: string; metadata: FilingCitation };
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

type JsonRecord = Record<string, unknown>;

function asRecord(value: unknown): JsonRecord | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonRecord
    : null;
}

function requiredString(value: unknown, field: string): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new ApiError(`The answer service returned an invalid ${field}.`, 502, "invalid_response");
  }
  return value;
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function optionalTrimmedString(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed || undefined;
}

function parseSourceUrl(value: unknown, required: boolean): string | undefined {
  const sourceUrl = optionalTrimmedString(value);
  if (!sourceUrl) {
    if (required) throw new ApiError("The answer service omitted the web citation URL.", 502, "invalid_response");
    return undefined;
  }
  try {
    if (
      sourceUrl.length > 4_096
      || [...sourceUrl].some((character) => (
        /\s/.test(character)
        || character.charCodeAt(0) < 32
        || character.charCodeAt(0) === 127
        || character === "\\"
      ))
    ) throw new Error("unsafe URL characters");
    const parsed = new URL(sourceUrl);
    if (
      (parsed.protocol !== "http:" && parsed.protocol !== "https:")
      || !parsed.hostname
      || parsed.username
      || parsed.password
    ) throw new Error("unsupported URL");
  } catch {
    throw new ApiError("The answer service returned an invalid citation URL.", 502, "invalid_response");
  }
  return sourceUrl;
}

function parseCitation(value: unknown): Citation {
  const item = asRecord(value);
  if (!item) throw new ApiError("The answer service returned an invalid citation.", 502, "invalid_response");
  const optionalScalar = (input: unknown) => typeof input === "string" || typeof input === "number" || input === null ? input : undefined;
  const kind = item.kind === undefined ? "filing" : item.kind;
  if (kind !== "filing" && kind !== "web") {
    throw new ApiError("The answer service returned an invalid citation kind.", 502, "invalid_response");
  }
  const common = {
    citation_id: requiredString(item.citation_id, "citation ID"),
    label: requiredString(item.label, "citation label"),
    title: optionalTrimmedString(item.title),
  };
  if (kind === "web") {
    return {
      ...common,
      kind,
      source_url: parseSourceUrl(item.source_url, true)!,
      target_chunk_id: optionalTrimmedString(item.target_chunk_id),
      ticker: optionalTrimmedString(item.ticker),
      record_type: optionalTrimmedString(item.record_type),
      report_date: optionalString(item.report_date),
      filing_date: optionalString(item.filing_date),
      section_title: optionalString(item.section_title),
      page_start: optionalScalar(item.page_start),
      page_end: optionalScalar(item.page_end),
      display_page_start: optionalScalar(item.display_page_start),
      display_page_end: optionalScalar(item.display_page_end),
    };
  }
  return {
    ...common,
    kind,
    target_chunk_id: requiredString(item.target_chunk_id, "citation target"),
    ticker: requiredString(item.ticker, "citation ticker"),
    record_type: requiredString(item.record_type, "citation record type"),
    report_date: optionalString(item.report_date),
    filing_date: optionalString(item.filing_date),
    section_title: optionalString(item.section_title),
    page_start: optionalScalar(item.page_start),
    page_end: optionalScalar(item.page_end),
    display_page_start: optionalScalar(item.display_page_start),
    display_page_end: optionalScalar(item.display_page_end),
    source_url: parseSourceUrl(item.source_url, false),
  };
}

function parseDiagnostics(value: unknown): Diagnostics | undefined {
  const item = asRecord(value);
  if (!item) return undefined;
  const rawStages = Array.isArray(item.stages) ? item.stages : [];
  const stages = rawStages.flatMap((stage) => {
    const parsed = asRecord(stage);
    if (!parsed || typeof parsed.stage !== "string") return [];
    return [{
      ...parsed,
      stage: parsed.stage,
      status: typeof parsed.status === "string" ? parsed.status : "unknown",
      latency_ms: typeof parsed.latency_ms === "number" ? parsed.latency_ms : undefined,
    }];
  });
  const rawPlans = Array.isArray(item.bank_plans) ? item.bank_plans : [];
  const bankPlans = rawPlans.flatMap((plan) => {
    const parsed = asRecord(plan);
    if (!parsed || typeof parsed.ticker !== "string" || typeof parsed.action !== "string") return [];
    return [parsed as Diagnostics["bank_plans"][number]];
  });
  const rawGate = asRecord(item.quality_gate);
  const rawChecks = asRecord(rawGate?.checks);
  const checks = Object.fromEntries(
    Object.entries(rawChecks ?? {}).filter((entry): entry is [string, boolean] => typeof entry[1] === "boolean"),
  );
  return {
    route: optionalTrimmedString(item.route) ?? "unknown",
    agentic_rag_enabled: item.agentic_rag_enabled === true,
    outcome: typeof item.outcome === "string" ? item.outcome : "unknown",
    failed_stage: typeof item.failed_stage === "string" || item.failed_stage === null ? item.failed_stage : undefined,
    error_code: typeof item.error_code === "string" || item.error_code === null ? item.error_code : undefined,
    stages,
    initial_evidence_count: typeof item.initial_evidence_count === "number" || item.initial_evidence_count === null ? item.initial_evidence_count : undefined,
    final_evidence_count: typeof item.final_evidence_count === "number" || item.final_evidence_count === null ? item.final_evidence_count : undefined,
    model_request_count: typeof item.model_request_count === "number" || item.model_request_count === null ? item.model_request_count : undefined,
    bank_plans: bankPlans,
    quality_gate: { passed: rawGate?.passed === true, checks },
  };
}

function parseBankResult(value: unknown): BankResult {
  const item = asRecord(value);
  if (!item) throw new ApiError("The answer service returned an invalid bank result.", 502, "invalid_response");
  const status = item.status;
  const answerType = item.answer_type;
  if (status !== "supported" && status !== "ambiguous" && status !== "unsupported") {
    throw new ApiError("The answer service returned an invalid bank status.", 502, "invalid_response");
  }
  if (answerType !== "numeric" && answerType !== "narrative") {
    throw new ApiError("The answer service returned an invalid answer type.", 502, "invalid_response");
  }
  if (!Array.isArray(item.citations)) {
    throw new ApiError("The answer service omitted bank citations.", 502, "invalid_response");
  }
  return {
    ticker: requiredString(item.ticker, "bank ticker"),
    bank_name: requiredString(item.bank_name, "bank name"),
    status,
    answer_type: answerType,
    answer: requiredString(item.answer, "bank answer"),
    facts: asRecord(item.facts) as NumericFacts | null,
    reason: requiredString(item.reason, "bank reason"),
    citations: item.citations.map(parseCitation),
  };
}

export function parseAnswerPayload(value: unknown): AnswerResponse {
  const item = asRecord(value);
  if (!item) throw new ApiError("The answer service returned an invalid answer.", 502, "invalid_response");
  const status = item.status;
  const answerType = item.answer_type;
  if (status !== "supported" && status !== "partial" && status !== "ambiguous" && status !== "unsupported") {
    throw new ApiError("The answer service returned an invalid answer status.", 502, "invalid_response");
  }
  if (answerType !== "numeric" && answerType !== "narrative") {
    throw new ApiError("The answer service returned an invalid answer type.", 502, "invalid_response");
  }
  if (!Array.isArray(item.citations)) {
    throw new ApiError("The answer service omitted citations.", 502, "invalid_response");
  }
  const allowedDialogActs: ReadonlySet<DialogAct> = new Set([
    "answer",
    "clarification",
    "greeting",
    "acknowledgement",
    "capability",
    "general_explanation",
    "out_of_scope",
    "retryable_error",
    "contextual_transform",
    "web_answer",
    "web_research_unavailable",
    "calculation",
  ]);
  const dialogAct = typeof item.dialog_act === "string" && allowedDialogActs.has(item.dialog_act as DialogAct)
    ? item.dialog_act as DialogAct
    : undefined;
  return {
    question: requiredString(item.question, "question"),
    dialog_act: dialogAct,
    mode: item.mode === "comparison" ? "comparison" : undefined,
    ticker: typeof item.ticker === "string" ? item.ticker : null,
    tickers: Array.isArray(item.tickers) ? item.tickers.filter((ticker): ticker is string => typeof ticker === "string") : undefined,
    status,
    answer_type: answerType,
    answer: requiredString(item.answer, "answer text"),
    reason: requiredString(item.reason, "answer reason"),
    citations: item.citations.map(parseCitation),
    bank_results: Array.isArray(item.bank_results) ? item.bank_results.map(parseBankResult) : undefined,
    diagnostics: parseDiagnostics(item.diagnostics),
  };
}

export function parseTurnPayload(value: unknown): Turn {
  const item = asRecord(value);
  if (!item) throw new ApiError("The answer service returned an invalid turn.", 502, "invalid_response");
  const state = item.state;
  if (state !== "loading" && state !== "answered" && state !== "error") {
    throw new ApiError("The answer service returned an invalid turn state.", 502, "invalid_response");
  }
  const turn: Turn = {
    id: requiredString(item.id, "turn ID"),
    question: requiredString(item.question, "turn question"),
    state,
    error: optionalString(item.error),
    error_code: optionalString(item.error_code),
    status: optionalString(item.status),
    diagnostics: parseDiagnostics(item.diagnostics),
    created_at: optionalString(item.created_at),
  };
  if (state === "answered") turn.response = parseAnswerPayload(item.response);
  if (state === "error" && !turn.error) turn.error = "The answer service reported an error.";
  return turn;
}

function parseThreadSummary(value: unknown): ThreadSummary {
  const item = asRecord(value);
  if (!item) throw new ApiError("The answer service returned an invalid conversation.", 502, "invalid_response");
  return {
    id: requiredString(item.id, "conversation ID"),
    title: requiredString(item.title, "conversation title"),
    session_ticker: typeof item.session_ticker === "string" ? item.session_ticker : null,
    session_tickers: Array.isArray(item.session_tickers) ? item.session_tickers.filter((ticker): ticker is string => typeof ticker === "string") : [],
    created_at: requiredString(item.created_at, "conversation creation time"),
    updated_at: requiredString(item.updated_at, "conversation update time"),
  };
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
  const payload = await requestJson<{ threads?: unknown }>("/api/threads", { signal });
  if (!Array.isArray(payload.threads)) throw new ApiError("The answer service omitted conversations.", 502, "invalid_response");
  return payload.threads.map(parseThreadSummary);
}

export async function createThread(title?: string): Promise<ThreadSummary> {
  const payload = await requestJson<unknown>("/api/threads", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: title || undefined }),
  });
  return parseThreadSummary(payload);
}

export async function loadThread(threadId: string, signal?: AbortSignal): Promise<ThreadHistory> {
  const payload = asRecord(await requestJson<unknown>(`/api/threads/${threadId}/messages`, { signal }));
  if (!payload || !Array.isArray(payload.turns)) throw new ApiError("The answer service returned invalid conversation history.", 502, "invalid_response");
  return { thread: parseThreadSummary(payload.thread), turns: payload.turns.map(parseTurnPayload) };
}

export async function renameThread(threadId: string, title: string): Promise<ThreadSummary> {
  const payload = await requestJson<unknown>(`/api/threads/${threadId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  return parseThreadSummary(payload);
}

export async function deleteThread(threadId: string): Promise<void> {
  const response = await fetch(`/api/threads/${threadId}`, { method: "DELETE" });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({})) as ErrorPayload;
    throw new ApiError(payload.detail || "Conversation could not be deleted.", response.status);
  }
}

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
  const contentType = response.headers.get("content-type");
  if (contentType && !contentType.toLowerCase().includes("text/event-stream")) {
    throw new ApiError("The answer service returned a non-streaming response.", 502, "invalid_stream");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalTurn: Turn | undefined;
  let malformedEvents = 0;

  const consumeBlock = (block: string) => {
    const data = block
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");
    if (!data || data === "[DONE]") return;
    try {
      const event = asRecord(JSON.parse(data));
      if (!event || typeof event.type !== "string") throw new Error("Invalid SSE event");
      if (event.type === "status" && typeof event.stage === "string") {
        onStatus(event.stage, typeof event.message === "string" ? event.message : "");
      }
      if (event.type === "answer" || event.type === "error") {
        finalTurn = parseTurnPayload(event.turn);
      }
    } catch {
      malformedEvents += 1;
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    let boundary = /\r?\n\r?\n/.exec(buffer);
    while (boundary) {
      consumeBlock(buffer.slice(0, boundary.index));
      buffer = buffer.slice(boundary.index + boundary[0].length);
      boundary = /\r?\n\r?\n/.exec(buffer);
    }
    if (done) break;
  }
  if (buffer.trim()) consumeBlock(buffer);
  if (!finalTurn) {
    const detail = malformedEvents ? ` (${malformedEvents} malformed event${malformedEvents === 1 ? "" : "s"})` : "";
    throw new ApiError(`The answer stream ended without a valid result${detail}.`, 502, "invalid_stream");
  }
  return finalTurn;
}

export async function loadCitationContext(
  citationId: string,
  signal?: AbortSignal,
): Promise<CitationContext> {
  const payload = asRecord(await requestJson<unknown>(`/api/citations/${citationId}/context?radius=1`, { signal }));
  if (!payload || !Array.isArray(payload.chunks)) throw new ApiError("The answer service returned invalid citation context.", 502, "invalid_response");
  const citationWrapper = asRecord(payload.citation);
  const metadata = parseCitation(asRecord(citationWrapper?.metadata));
  if (metadata.kind !== "filing") {
    throw new ApiError("The answer service returned invalid filing citation context.", 502, "invalid_response");
  }
  const chunks = payload.chunks.map((value) => {
    const chunk = asRecord(value);
    if (!chunk || (chunk.role !== "previous" && chunk.role !== "anchor" && chunk.role !== "next")) {
      throw new ApiError("The answer service returned an invalid source chunk.", 502, "invalid_response");
    }
    return {
      target_chunk_id: requiredString(chunk.target_chunk_id, "source target"),
      role: chunk.role,
      record_type: requiredString(chunk.record_type, "source record type"),
      document: requiredString(chunk.document, "source document"),
      metadata: asRecord(chunk.metadata) ?? {},
    } as SourceChunk;
  });
  return {
    citation: {
      id: requiredString(citationWrapper?.id, "stored citation ID"),
      label: requiredString(citationWrapper?.label, "stored citation label"),
      metadata,
    },
    target_chunk_id: requiredString(payload.target_chunk_id, "source target"),
    record_type: requiredString(payload.record_type, "source record type"),
    ticker: requiredString(payload.ticker, "source ticker"),
    source_url: typeof payload.source_url === "string" ? payload.source_url : "",
    corpus_hash: requiredString(payload.corpus_hash, "source corpus hash"),
    chunks,
  };
}

// Document types
export type UserDocument = {
  id: string;
  thread_id: string | null;
  filename: string;
  content_type: string;
  file_size: number;
  uploaded_at: string;
  metadata: Record<string, unknown>;
};

export type DocumentUploadResponse = {
  document: UserDocument;
};

export type DocumentListResponse = {
  documents: UserDocument[];
};

export type DocumentResponse = {
  document: UserDocument;
};

export type DocumentContentResponse = {
  content: string;
  document: UserDocument;
};

// Document API functions
export async function uploadDocument(file: File, threadId?: string): Promise<UserDocument> {
  const formData = new FormData();
  formData.append("file", file);
  if (threadId) {
    formData.append("thread_id", threadId);
  }

  const response = await fetch("/api/documents/upload", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({})) as ErrorPayload;
    throw new ApiError(
      payload.error || payload.detail || "Failed to upload document.",
      response.status,
      payload.code,
    );
  }

  const payload = await response.json() as DocumentUploadResponse;
  return payload.document;
}

export async function listDocuments(threadId?: string, signal?: AbortSignal): Promise<UserDocument[]> {
  const url = threadId ? `/api/documents?thread_id=${encodeURIComponent(threadId)}` : "/api/documents";
  const payload = await requestJson<DocumentListResponse>(url, { signal });
  return payload.documents;
}

export async function getDocument(documentId: string, signal?: AbortSignal): Promise<UserDocument> {
  const payload = await requestJson<DocumentResponse>(`/api/documents/${documentId}`, { signal });
  return payload.document;
}

export async function getDocumentContent(documentId: string, signal?: AbortSignal): Promise<DocumentContentResponse> {
  const payload = await requestJson<DocumentContentResponse>(`/api/documents/${documentId}/content`, { signal });
  return payload;
}

export async function deleteDocument(documentId: string): Promise<void> {
  const response = await fetch(`/api/documents/${documentId}`, { method: "DELETE" });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({})) as ErrorPayload;
    throw new ApiError(
      payload.detail || "Failed to delete document.",
      response.status,
      payload.code,
    );
  }
}
