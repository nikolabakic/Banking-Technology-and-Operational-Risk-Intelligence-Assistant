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

    fireEvent.pointerDown(screen.getByRole("button", { name: `Conversation actions for ${thread.title}` }), { button: 0, ctrlKey: false });
    fireEvent.click(await screen.findByRole("menuitem", { name: "Rename" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Conversation title" }), { target: { value: "Renamed" } });
    fireEvent.click(screen.getByRole("button", { name: "Save title" }));
    await waitFor(() => expect(mocks.renameThread).toHaveBeenCalledWith(thread.id, "Renamed"));

    fireEvent.pointerDown(await screen.findByRole("button", { name: "Conversation actions for Renamed" }), { button: 0, ctrlKey: false });
    fireEvent.click(await screen.findByRole("menuitem", { name: "Delete" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete conversation" }));
    await waitFor(() => expect(mocks.deleteThread).toHaveBeenCalledWith(thread.id));
  });
});
