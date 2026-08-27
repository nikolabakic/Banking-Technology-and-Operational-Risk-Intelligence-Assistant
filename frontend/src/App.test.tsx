import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, useNavigate } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import ErrorBoundary from "./ErrorBoundary";
import type { AnswerResponse, CitationContext, FilingCitation, ThreadSummary, Turn } from "./api";

const mocks = vi.hoisted(() => ({
  listThreads: vi.fn(),
  loadThread: vi.fn(),
  renameThread: vi.fn(),
  deleteThread: vi.fn(),
  loadCitationContext: vi.fn(),
  createThread: vi.fn(),
  streamAnswer: vi.fn(),
  uploadDocument: vi.fn(),
}));

vi.mock("./api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./api")>()),
  ...mocks,
}));

const thread: ThreadSummary = {
  id: "11111111-1111-4111-8111-111111111111",
  title: "JPM operational risk",
  session_ticker: "JPM",
  session_tickers: ["JPM"],
  created_at: "2026-08-12T10:00:00Z",
  updated_at: "2026-08-12T10:01:00Z",
};

const filingCitation: FilingCitation = {
  kind: "filing",
  citation_id: "22222222-2222-4222-8222-222222222222",
  label: "E1",
  target_chunk_id: "chunk-1",
  ticker: "JPM",
  record_type: "text",
  section_title: "Risk management",
};

const response: AnswerResponse = {
  question: "How does JPM describe operational risk?",
  ticker: "JPM",
  status: "supported",
  answer_type: "narrative",
  answer: "JPM uses a risk framework [E1]",
  reason: "Supported",
  citations: [filingCitation],
  diagnostics: {
    route: "domain_rag",
    agentic_rag_enabled: true,
    outcome: "supported",
    stages: [{ stage: "routing", status: "completed", latency_ms: 12.3 }],
    initial_evidence_count: 5,
    final_evidence_count: 5,
    model_request_count: 3,
    bank_plans: [{
      ticker: "JPM",
      action: "generate",
      reason_code: "evidence_sufficient",
      explanation: "The initial evidence directly supports the answer.",
    }],
    quality_gate: { passed: true, checks: { pipeline_completed: true } },
  },
};

const turn: Turn = {
  id: "33333333-3333-4333-8333-333333333333",
  question: response.question,
  state: "answered",
  response,
};

const context: CitationContext = {
  citation: { id: filingCitation.citation_id, label: "E1", metadata: filingCitation },
  target_chunk_id: "chunk-1",
  record_type: "text",
  ticker: "JPM",
  source_url: "https://example.com/filing",
  corpus_hash: "hash-1",
  chunks: [{
    target_chunk_id: "chunk-1",
    role: "anchor",
    record_type: "text",
    document: "Canonical operational risk evidence.",
    metadata: { section_title: "Risk management", page_start: 42 },
  }],
};

function RouteControls() {
  const navigate = useNavigate();
  return <button type="button" onClick={() => navigate("/")}>Go to empty workspace</button>;
}

function renderThread() {
  return render(
    <MemoryRouter initialEntries={[`/chats/${thread.id}`]}>
      <App />
    </MemoryRouter>,
  );
}

describe("persistent chat workspace", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.listThreads.mockResolvedValue([thread]);
    mocks.loadThread.mockResolvedValue({ thread, turns: [turn] });
    mocks.renameThread.mockResolvedValue({ ...thread, title: "Renamed" });
    mocks.deleteThread.mockResolvedValue(undefined);
    mocks.loadCitationContext.mockResolvedValue(context);
    mocks.streamAnswer.mockResolvedValue(turn);
    mocks.uploadDocument.mockResolvedValue(undefined);
  });

  it("restores a saved thread and opens its canonical citation source", async () => {
    const { container } = renderThread();
    expect(await screen.findByText(response.question)).toBeInTheDocument();
    expect(container.querySelector(".assistant-mark img")).toHaveAttribute("src", "/brand/bankscope-target.svg");
    fireEvent.click(screen.getByRole("button", { name: "E1" }));
    expect(await screen.findByText("Canonical operational risk evidence.")).toBeInTheDocument();
    expect(mocks.loadCitationContext).toHaveBeenCalledWith(
      response.citations[0].citation_id,
      expect.any(AbortSignal),
    );
  });

  it("keeps execution diagnostics collapsed until requested", async () => {
    renderThread();
    await screen.findByText(response.question);
    const trigger = screen.getByRole("button", { name: "Diagnostics" });
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(trigger.closest(".answer-actions")).not.toBeNull();
    fireEvent.click(trigger);
    expect(await screen.findByText("Execution checks: passed")).toBeInTheDocument();
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("5 → 5")).toBeInTheDocument();
    expect(screen.getByText("✓")).toBeInTheDocument();
  });

  it("renders legacy error turns with empty diagnostics instead of crashing", async () => {
    mocks.loadThread.mockResolvedValue({
      thread,
      turns: [{
        id: "55555555-5555-4555-8555-555555555555",
        question: "A failed question",
        state: "error",
        error: "The answer failed.",
        diagnostics: {},
      } as Turn],
    });

    renderThread();

    expect(await screen.findByText("The answer failed.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Diagnostics" }));
    expect(await screen.findByText("Execution checks: failed")).toBeInTheDocument();
  });

  it("renders clarification as a normal assistant turn without fake source metadata", async () => {
    const clarification: AnswerResponse = {
      ...response,
      dialog_act: "clarification",
      ticker: null,
      status: "ambiguous",
      answer: "Which bank should I research?",
      reason: "A bank is required.",
      citations: [],
    };
    mocks.loadThread.mockResolvedValue({
      thread,
      turns: [{ ...turn, response: clarification }],
    });

    renderThread();

    expect(await screen.findByText("Which bank should I research?")).toBeInTheDocument();
    expect(screen.getByText("One detail is needed")).toBeInTheDocument();
    expect(screen.getByText("Clarification")).toBeInTheDocument();
    expect(screen.queryByText("0 sources")).not.toBeInTheDocument();
    expect(screen.queryByText("Grounded in indexed filings")).not.toBeInTheDocument();
  });

  it("labels mixed document and filing answers with both source classes", async () => {
    const mixed: AnswerResponse = {
      ...response,
      source_scope: "uploaded_document_and_indexed_filing",
      diagnostics: { ...response.diagnostics!, route: "document_filing_comparison" },
      citations: [
        {
          kind: "document",
          citation_id: "44444444-4444-4444-8444-444444444444",
          label: "E1",
          target_chunk_id: "user_document:doc-1",
          ticker: "UPLOAD",
          record_type: "text",
          document_id: "doc-1",
          filename: "north-river.pdf",
        },
      ],
    };
    mocks.loadThread.mockResolvedValue({ thread, turns: [{ ...turn, response: mixed }] });

    renderThread();

    expect(
      await screen.findByText("Grounded in uploaded document and indexed filing"),
    ).toBeInTheDocument();
  });

  it("retries a retryable answer through the existing stream path only after the user clicks", async () => {
    const retryQuestion = "How does Ally define operational risk?";
    const retryableResponse: AnswerResponse = {
      ...response,
      question: retryQuestion,
      dialog_act: "retryable_error",
      ticker: null,
      status: "unsupported",
      answer: "I couldn't complete reliable research for this message.",
      reason: "Research failed safely.",
      citations: [],
    };
    mocks.loadThread.mockResolvedValue({
      thread,
      turns: [{ ...turn, question: retryQuestion, response: retryableResponse }],
    });
    mocks.streamAnswer.mockReturnValue(new Promise<Turn>(() => undefined));

    renderThread();

    const retryButton = await screen.findByRole("button", { name: "Retry" });
    expect(screen.getAllByText(retryQuestion)).toHaveLength(1);
    expect(mocks.streamAnswer).not.toHaveBeenCalled();
    expect(retryButton).toBeEnabled();

    fireEvent.click(retryButton);

    await waitFor(() => expect(mocks.streamAnswer).toHaveBeenCalledWith(
      thread.id,
      retryQuestion,
      expect.any(Function),
      expect.any(AbortSignal),
    ));
    expect(screen.getAllByText(retryQuestion)).toHaveLength(2);
    expect(retryButton).toBeDisabled();
  });

  it("guards a new conversation before asynchronous thread creation completes", async () => {
    mocks.createThread.mockReturnValue(new Promise<ThreadSummary>(() => undefined));
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>,
    );
    const composer = await screen.findByPlaceholderText(
      "Ask naturally, research a filing, search the web, or calculate…",
    );
    fireEvent.change(composer, { target: { value: "Hello BankScope" } });
    const form = composer.closest("form");
    expect(form).not.toBeNull();

    fireEvent.submit(form!);
    fireEvent.submit(form!);

    expect(mocks.createThread).toHaveBeenCalledTimes(1);
    expect(mocks.streamAnswer).not.toHaveBeenCalled();
  });

  it("preserves a newly created conversation when the initial thread list resolves late", async () => {
    const newThread = { ...thread, id: "77777777-7777-4777-8777-777777777777", title: "New fast thread" };
    let resolveThreads!: (threads: ThreadSummary[]) => void;
    mocks.listThreads.mockReturnValue(new Promise((resolve) => { resolveThreads = resolve; }));
    mocks.createThread.mockResolvedValue(newThread);
    mocks.streamAnswer.mockReturnValue(new Promise<Turn>(() => undefined));
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>,
    );

    const composer = screen.getByRole("textbox", { name: "Research question" });
    fireEvent.change(composer, { target: { value: "Create immediately" } });
    fireEvent.submit(composer.closest("form")!);
    await waitFor(() => expect(mocks.streamAnswer).toHaveBeenCalledWith(
      newThread.id,
      "Create immediately",
      expect.any(Function),
      expect.any(AbortSignal),
    ));

    resolveThreads([thread]);

    expect(await screen.findByTitle(newThread.title)).toBeInTheDocument();
    expect(await screen.findByTitle(thread.title)).toBeInTheDocument();
  });

  it("removes a conversation created for an upload when the upload fails", async () => {
    const uploadThread = { ...thread, id: "88888888-8888-4888-8888-888888888888", title: "File: broken.pdf" };
    mocks.createThread.mockResolvedValue(uploadThread);
    mocks.uploadDocument.mockRejectedValue(new Error("Upload failed safely"));
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Upload file" }));
    const input = await screen.findByLabelText("Choose document to upload");
    fireEvent.change(input, {
      target: { files: [new File(["broken"], "broken.pdf", { type: "application/pdf" })] },
    });
    fireEvent.click(screen.getByRole("button", { name: "Upload File" }));

    await waitFor(() => expect(mocks.deleteThread).toHaveBeenCalledWith(uploadThread.id));
    expect(await screen.findByText("Upload failed safely")).toBeInTheDocument();
    expect(screen.queryByTitle(uploadThread.title)).not.toBeInTheDocument();
  });

  it("does not carry a previous conversation into a thread created after returning home", async () => {
    const newThread = { ...thread, id: "66666666-6666-4666-8666-666666666666", title: "New conversation" };
    mocks.createThread.mockResolvedValue(newThread);
    mocks.streamAnswer.mockReturnValue(new Promise<Turn>(() => undefined));
    render(
      <MemoryRouter initialEntries={[`/chats/${thread.id}`]}>
        <RouteControls />
        <App />
      </MemoryRouter>,
    );

    expect(await screen.findByText(response.question)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Go to empty workspace" }));
    await waitFor(() => expect(screen.queryByText(response.question)).not.toBeInTheDocument());

    const composer = screen.getByRole("textbox", { name: "Research question" });
    fireEvent.change(composer, { target: { value: "Start clean" } });
    fireEvent.submit(composer.closest("form")!);

    await waitFor(() => expect(mocks.streamAnswer).toHaveBeenCalledWith(
      newThread.id,
      "Start clean",
      expect.any(Function),
      expect.any(AbortSignal),
    ));
    expect(screen.queryByText(response.question)).not.toBeInTheDocument();
    expect(await screen.findByText("Start clean")).toBeInTheDocument();
  });

  it("ignores a previous thread load that resolves after navigation returned home", async () => {
    let resolveHistory!: (history: { thread: ThreadSummary; turns: Turn[] }) => void;
    mocks.loadThread.mockReturnValue(new Promise((resolve) => { resolveHistory = resolve; }));
    render(
      <MemoryRouter initialEntries={[`/chats/${thread.id}`]}>
        <RouteControls />
        <App />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Go to empty workspace" }));
    resolveHistory({ thread, turns: [turn] });

    expect(await screen.findByRole("textbox", { name: "Research question" })).toBeInTheDocument();
    expect(screen.queryByText(response.question)).not.toBeInTheDocument();
  });

  it("renders web citations as safe external links without opening the filing evidence viewer", async () => {
    const webResponse: AnswerResponse = {
      ...response,
      dialog_act: "web_answer",
      ticker: null,
      answer: "The latest official update is available here [E1]",
      citations: [{
        kind: "web",
        citation_id: "web-result-1",
        label: "E1",
        title: "Official risk update",
        source_url: "https://example.com/risk-update",
      }],
    };
    mocks.loadThread.mockResolvedValue({
      thread,
      turns: [{ ...turn, response: webResponse }],
    });

    renderThread();

    const citationLink = await screen.findByRole("link", { name: "E1" });
    expect(citationLink).toHaveAttribute("href", "https://example.com/risk-update");
    expect(citationLink).toHaveAttribute("target", "_blank");
    expect(citationLink).toHaveAttribute("rel", "noopener noreferrer");
    expect(citationLink).toHaveAttribute("title", "Open source E1: Official risk update");
    expect(screen.getByText("Researched on the web")).toBeInTheDocument();
    expect(screen.getByText("Web research")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "E1" })).not.toBeInTheDocument();
    expect(mocks.loadCitationContext).not.toHaveBeenCalled();
  });

  it("renames and deletes conversations through confirmed sidebar actions", async () => {
    renderThread();
    await screen.findByText(response.question);

    fireEvent.click(screen.getByRole("button", { name: `Rename ${thread.title}` }));
    fireEvent.change(screen.getByRole("textbox", { name: "Conversation title" }), { target: { value: "Renamed" } });
    fireEvent.click(screen.getByRole("button", { name: "Save title" }));
    await waitFor(() => expect(mocks.renameThread).toHaveBeenCalledWith(thread.id, "Renamed"));

    fireEvent.click(await screen.findByRole("button", { name: "Delete Renamed" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete conversation" }));
    await waitFor(() => expect(mocks.deleteThread).toHaveBeenCalledWith(thread.id));
  });

  it("keeps actions available when a generated conversation title is very long", async () => {
    const longTitle = "What are the most important operational, technology, cyber security, third-party and resilience risks disclosed by JPMorgan Chase?";
    mocks.listThreads.mockResolvedValue([{ ...thread, title: longTitle }]);
    mocks.loadThread.mockResolvedValue({ thread: { ...thread, title: longTitle }, turns: [turn] });

    renderThread();
    await screen.findByText(response.question);

    fireEvent.click(screen.getByRole("button", { name: `Rename ${longTitle}` }));
    expect(screen.getByRole("textbox", { name: "Conversation title" })).toHaveValue(longTitle);
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    fireEvent.click(screen.getByRole("button", { name: `Delete ${longTitle}` }));
    expect(screen.getByRole("button", { name: "Delete conversation" })).toBeInTheDocument();
  });

  it("renders markdown tables semantically and keeps citations interactive", async () => {
    const tableResponse = {
      ...response,
      answer: "| Risk | Description |\n| --- | --- |\n| Operational | Failed processes [E1] |\n| Cyber | Security incidents |",
    };
    mocks.loadThread.mockResolvedValue({
      thread,
      turns: [{ ...turn, response: tableResponse }],
    });

    renderThread();

    const table = await screen.findByRole("table");
    const tableRegion = screen.getByRole("region", { name: "Scrollable data table" });
    expect(table).toHaveTextContent("Risk");
    expect(table).toHaveTextContent("Operational");
    expect(tableRegion).toContainElement(table);
    expect(tableRegion).toHaveAttribute("tabindex", "0");
    expect(screen.getAllByRole("row")).toHaveLength(3);
    fireEvent.click(screen.getByRole("button", { name: "E1" }));
    expect(await screen.findByText("Canonical operational risk evidence.")).toBeInTheDocument();
  });

  it("renders filing tables in the evidence viewer instead of raw markdown", async () => {
    mocks.loadCitationContext.mockResolvedValue({
      ...context,
      chunks: [{
        ...context.chunks[0],
        document: "**JPMorgan Chase & Co.**\n\n**Consolidated balance sheets**\n\n| December 31, (in millions) | 2025 | 2024 |\n| --- | --- | --- |\n| Total assets | $ 4,424,900 | $ 4,002,814 |",
      }],
    });

    renderThread();
    await screen.findByText(response.question);
    fireEvent.click(screen.getByRole("button", { name: "E1" }));

    expect(await screen.findByRole("table")).toHaveTextContent("Total assets");
    expect(screen.getByText("JPMorgan Chase & Co.").tagName).toBe("STRONG");
    expect(screen.queryByText("| --- | --- | --- |", { exact: true })).not.toBeInTheDocument();
  });

  it("renders a comparison summary, bank cards, partial status and bank sources", async () => {
    const bacCitation = {
      ...filingCitation,
      citation_id: "44444444-4444-4444-8444-444444444444",
      label: "E2",
      ticker: "BAC",
      target_chunk_id: "chunk-2",
    };
    const comparisonResponse: AnswerResponse = {
      ...response,
      mode: "comparison",
      ticker: null,
      tickers: ["JPM", "BAC"],
      status: "partial",
      answer: "JPM is supported [E1]; BAC lacks sufficient evidence.",
      citations: [filingCitation, bacCitation],
      bank_results: [
        {
          ticker: "JPM",
          bank_name: "JPMorgan Chase & Co.",
          status: "supported",
          answer_type: "narrative",
          answer: "JPM uses a risk framework [E1]",
          facts: null,
          reason: "Supported",
          citations: [response.citations[0]],
        },
        {
          ticker: "BAC",
          bank_name: "Bank of America Corporation",
          status: "unsupported",
          answer_type: "narrative",
          answer: "Insufficient evidence.",
          facts: null,
          reason: "No evidence",
          citations: [],
        },
      ],
    };
    mocks.loadThread.mockResolvedValue({
      thread: { ...thread, session_ticker: null, session_tickers: ["JPM", "BAC"] },
      turns: [{ ...turn, response: comparisonResponse }],
    });

    renderThread();

    expect(await screen.findByText("JPMorgan Chase & Co.")).toBeInTheDocument();
    expect(screen.getByText("Bank of America Corporation")).toBeInTheDocument();
    expect(screen.getAllByText("partial")).toHaveLength(1);
    expect(screen.getByText("JPM vs BAC")).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole("button", { name: "E1" })[0]);
    expect(await screen.findByText("Canonical operational risk evidence.")).toBeInTheDocument();
  });

  it("maps every answer status to a consistent semantic badge", async () => {
    const statusResponse: AnswerResponse = {
      ...response,
      mode: "comparison",
      ticker: null,
      tickers: ["JPM", "BAC", "C"],
      status: "partial",
      bank_results: [
        {
          ticker: "JPM",
          bank_name: "JPMorgan Chase & Co.",
          status: "supported",
          answer_type: "narrative",
          answer: "Supported answer.",
          facts: null,
          reason: "Supported",
          citations: [],
        },
        {
          ticker: "BAC",
          bank_name: "Bank of America Corporation",
          status: "ambiguous",
          answer_type: "narrative",
          answer: "Ambiguous answer.",
          facts: null,
          reason: "Ambiguous",
          citations: [],
        },
        {
          ticker: "C",
          bank_name: "Citigroup Inc.",
          status: "unsupported",
          answer_type: "narrative",
          answer: "Unsupported answer.",
          facts: null,
          reason: "Unsupported",
          citations: [],
        },
      ],
    };
    mocks.loadThread.mockResolvedValue({
      thread: { ...thread, session_ticker: null, session_tickers: ["JPM", "BAC", "C"] },
      turns: [{ ...turn, response: statusResponse }],
    });

    const { container } = renderThread();
    await screen.findByText("Citigroup Inc.");

    expect(container.querySelector('[data-status="supported"]')).toHaveClass("ui-badge-success");
    expect(container.querySelector('[data-status="partial"]')).toHaveClass("ui-badge-warning");
    expect(container.querySelector('[data-status="ambiguous"]')).toHaveClass("ui-badge-warning");
    expect(container.querySelector('[data-status="unsupported"]')).toHaveClass("ui-badge-danger");
  });

  it("keeps new conversation and corpus status only in the header", async () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>,
    );

    await waitFor(() => expect(mocks.listThreads).toHaveBeenCalled());
    expect(screen.getAllByRole("button", { name: /New conversation/i })).toHaveLength(1);
    expect(screen.getAllByText("Corpus ready")).toHaveLength(1);
    expect(screen.getByRole("img", { name: "BankScope" })).toHaveAttribute("src", "/brand/bankscope-wordmark.svg");
    expect(screen.getByText("BankScope")).toBeInTheDocument();
    expect(screen.queryByText("Banking risk intelligence")).not.toBeInTheDocument();
    expect(screen.queryByText("Evidence corpus ready")).not.toBeInTheDocument();
  });
});

describe("render recovery", () => {
  it("shows a reload action when an unexpected child render fails", () => {
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const Broken = () => {
      throw new Error("invalid payload reached render");
    };

    try {
      render(<ErrorBoundary><Broken /></ErrorBoundary>);
      expect(screen.getByRole("alert")).toHaveTextContent("interface recovered");
      expect(screen.getByRole("button", { name: "Reload BankScope" })).toBeInTheDocument();
    } finally {
      consoleSpy.mockRestore();
    }
  });
});
