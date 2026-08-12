import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { AnswerResponse, CitationContext, ThreadSummary, Turn } from "./api";

const mocks = vi.hoisted(() => ({
  listThreads: vi.fn(),
  loadThread: vi.fn(),
  renameThread: vi.fn(),
  deleteThread: vi.fn(),
  loadCitationContext: vi.fn(),
  createThread: vi.fn(),
  streamAnswer: vi.fn(),
}));

vi.mock("./api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./api")>()),
  ...mocks,
}));

const thread: ThreadSummary = {
  id: "11111111-1111-4111-8111-111111111111",
  title: "JPM operational risk",
  session_ticker: "JPM",
  created_at: "2026-08-12T10:00:00Z",
  updated_at: "2026-08-12T10:01:00Z",
};

const response: AnswerResponse = {
  question: "How does JPM describe operational risk?",
  ticker: "JPM",
  status: "supported",
  answer_type: "narrative",
  answer: "JPM uses a risk framework [E1]",
  reason: "Supported",
  citations: [{
    citation_id: "22222222-2222-4222-8222-222222222222",
    label: "E1",
    target_chunk_id: "chunk-1",
    ticker: "JPM",
    record_type: "text",
    section_title: "Risk management",
  }],
};

const turn: Turn = {
  id: "33333333-3333-4333-8333-333333333333",
  question: response.question,
  state: "answered",
  response,
};

const context: CitationContext = {
  citation: { id: response.citations[0].citation_id, label: "E1", metadata: response.citations[0] },
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
  });

  it("restores a saved thread and opens its canonical citation source", async () => {
    renderThread();
    expect(await screen.findByText(response.question)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "E1" }));
    expect(await screen.findByText("Canonical operational risk evidence.")).toBeInTheDocument();
    expect(mocks.loadCitationContext).toHaveBeenCalledWith(
      response.citations[0].citation_id,
      expect.any(AbortSignal),
    );
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
    expect(table).toHaveTextContent("Risk");
    expect(table).toHaveTextContent("Operational");
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

  it("keeps new conversation and corpus status only in the header", async () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>,
    );

    await waitFor(() => expect(mocks.listThreads).toHaveBeenCalled());
    expect(screen.getAllByRole("button", { name: /New conversation/i })).toHaveLength(1);
    expect(screen.getAllByText("Corpus ready")).toHaveLength(1);
    expect(screen.getByText("BankScope")).toBeInTheDocument();
    expect(screen.queryByText("Banking risk intelligence")).not.toBeInTheDocument();
    expect(screen.queryByText("Evidence corpus ready")).not.toBeInTheDocument();
  });
});
