import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, parseAnswerPayload, parseTurnPayload, streamAnswer } from "./api";

const answer = {
  question: "How does JPMorgan describe operational risk?",
  ticker: "JPM",
  status: "supported",
  answer_type: "narrative",
  answer: "The filing describes operational risk.",
  reason: "Supported.",
  citations: [],
};

const turn = {
  id: "33333333-3333-4333-8333-333333333333",
  question: answer.question,
  state: "answered",
  response: answer,
};

const legacyFilingCitation = {
  citation_id: "22222222-2222-4222-8222-222222222222",
  label: "E1",
  target_chunk_id: "chunk-1",
  ticker: "JPM",
  record_type: "text",
};

function streamResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)));
      controller.close();
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "Content-Type": "text/event-stream; charset=utf-8" },
  });
}

describe("answer API contracts", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("parses fragmented CRLF SSE events and ignores heartbeat comments", async () => {
    const payload = JSON.stringify({ type: "answer", turn });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamResponse([
      ": keep-alive\r\n\r\ndata: {\"type\":\"status\",\"stage\":\"retr",
      "ieving\",\"message\":\"Searching\"}\r\n\r\ndata: ",
      `${payload}\r\n\r\ndata: {"type":"done","status":200}\r\n\r\n`,
    ])));
    const statuses: string[] = [];

    const result = await streamAnswer("thread-1", answer.question, (stage) => statuses.push(stage));

    expect(result.response?.answer).toBe(answer.answer);
    expect(statuses).toEqual(["retrieving"]);
  });

  it("contains a malformed event and still accepts the next valid final event", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamResponse([
      "data: definitely-not-json\n\n",
      `data: ${JSON.stringify({ type: "answer", turn })}\n\n`,
    ])));

    await expect(streamAnswer("thread-1", answer.question, () => undefined)).resolves.toMatchObject({
      state: "answered",
    });
  });

  it("rejects a stream with no valid final turn without throwing into React render", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamResponse([
      "data: {broken\n\n",
    ])));

    await expect(streamAnswer("thread-1", answer.question, () => undefined)).rejects.toMatchObject({
      code: "invalid_stream",
    });
  });

  it("rejects malformed answers and normalizes legacy empty diagnostics", () => {
    expect(() => parseAnswerPayload({ ...answer, citations: undefined })).toThrow(ApiError);
    const errorTurn = parseTurnPayload({
      id: "turn-error",
      question: "Question",
      state: "error",
      error: "Failed",
      diagnostics: {},
    });

    expect(errorTurn.diagnostics?.quality_gate).toEqual({ passed: false, checks: {} });
    expect(parseAnswerPayload(answer).evidence_audit).toBeUndefined();
    expect(parseAnswerPayload({
      ...answer,
      evidence_audit: { status: "passed", grounded: true },
    }).evidence_audit).toBeUndefined();
  });

  it("parses a valid evidence audit without making it part of answer validity", () => {
    const parsed = parseAnswerPayload({
      ...answer,
      evidence_audit: {
        status: "review_recommended",
        question_addressed: true,
        grounded: false,
        citation_coverage_ok: false,
        contradiction_found: false,
        summary: "One material claim needs review.",
        metadata: { model: "judge-model", prompt_version: "runtime-v1" },
      },
    });

    expect(parsed.evidence_audit).toMatchObject({
      status: "review_recommended",
      grounded: false,
      summary: "One material claim needs review.",
    });
  });

  it("parses the conversational dialog act without requiring citations", () => {
    expect(parseAnswerPayload({
      ...answer,
      ticker: null,
      dialog_act: "clarification",
      status: "ambiguous",
      answer: "Which bank do you mean?",
    }).dialog_act).toBe("clarification");
  });

  it("defaults legacy citations without a kind to filing citations", () => {
    const parsed = parseAnswerPayload({ ...answer, citations: [legacyFilingCitation] });

    expect(parsed.citations[0]).toMatchObject({
      kind: "filing",
      target_chunk_id: "chunk-1",
      ticker: "JPM",
      record_type: "text",
    });
  });

  it("accepts web citations without filing-only metadata and normalizes their title and URL", () => {
    const parsed = parseAnswerPayload({
      ...answer,
      dialog_act: "web_answer",
      citations: [{
        kind: "web",
        citation_id: "web-result-1",
        label: "E1",
        title: "  Official risk update  ",
        source_url: "  https://example.com/risk-update  ",
      }],
    });

    expect(parsed.citations[0]).toEqual({
      kind: "web",
      citation_id: "web-result-1",
      label: "E1",
      title: "Official risk update",
      source_url: "https://example.com/risk-update",
      target_chunk_id: undefined,
      ticker: undefined,
      record_type: undefined,
      report_date: undefined,
      filing_date: undefined,
      section_title: undefined,
      page_start: undefined,
      page_end: undefined,
      display_page_start: undefined,
      display_page_end: undefined,
    });
  });

  it.each([
    "javascript:alert(1)",
    "not-a-url",
    "file:///tmp/source.txt",
    "https://user:password@example.com/source",
    "https://example.com\\@unsafe.test/source",
    "https://example.com/source\u0000suffix",
  ])(
    "rejects an unsafe web citation URL: %s",
    (sourceUrl) => {
      expect(() => parseAnswerPayload({
        ...answer,
        citations: [{
          kind: "web",
          citation_id: "web-result-1",
          label: "E1",
          source_url: sourceUrl,
        }],
      })).toThrow(ApiError);
    },
  );

  it.each([
    "contextual_transform",
    "web_answer",
    "web_research_unavailable",
    "calculation",
  ] as const)("preserves the %s dialog act", (dialogAct) => {
    expect(parseAnswerPayload({ ...answer, dialog_act: dialogAct }).dialog_act).toBe(dialogAct);
  });

  it.each(["web_search", "calculator", "scope_guard"])(
    "preserves the %s diagnostics route",
    (route) => {
      const parsed = parseAnswerPayload({ ...answer, diagnostics: { route } });
      expect(parsed.diagnostics?.route).toBe(route);
    },
  );
});
