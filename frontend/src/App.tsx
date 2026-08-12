import { type FormEvent, type KeyboardEvent, useEffect, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowRight,
  BadgeCheck,
  ChevronLeft,
  ChevronRight,
  Copy,
  FileSearch,
  Menu,
  MessageSquare,
  Pencil,
  Plus,
  Send,
  Sparkles,
  Square,
  Trash2,
} from "lucide-react";
import { useMatch, useNavigate } from "react-router-dom";
import {
  ApiError,
  createThread,
  deleteThread,
  listThreads,
  loadCitationContext,
  loadThread,
  renameThread,
  streamAnswer,
  type AnswerResponse,
  type CitationContext,
  type ThreadSummary,
  type Turn,
} from "./api";
import { prompts } from "./data";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

type OpenSource = { response: AnswerResponse; index: number };

const stageLabels: Record<string, string> = {
  resolving_bank: "Identifying the bank…",
  embedding: "Encoding the question…",
  retrieving: "Searching indexed filings…",
  generating: "Generating a grounded answer…",
  validating: "Validating citations and answer structure…",
};

function Brand() {
  return (
    <div className="brand" aria-label="BankScope home">
      <span className="brand-mark">B</span>
      <span>bankscope</span>
    </div>
  );
}

function Composer({ value, onChange, onSubmit, onStop, loading, compact = false }: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (question?: string) => void;
  onStop: () => void;
  loading: boolean;
  compact?: boolean;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, compact ? 168 : 190)}px`;
  }, [compact, value]);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (value.trim() && !loading) onSubmit();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (value.trim() && !loading) onSubmit();
    }
  };

  return (
    <form className={`composer ${compact ? "composer-compact" : ""}`} onSubmit={submit}>
      <Textarea
        ref={textareaRef}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask anything about an indexed bank’s 10-K filing…"
        rows={compact ? 1 : 3}
        aria-label="Research question"
        disabled={loading}
      />
      <div className="composer-footer">
        <span><Sparkles size={13} /> Bank and filing detection is automatic</span>
        <div className="send-group">
          {!compact && <small>Enter to send · Shift + Enter for a new line</small>}
          {loading ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button type="button" size="icon" className="send-button stop-button" onClick={onStop} aria-label="Stop generating"><Square size={14} fill="currentColor" /></Button>
              </TooltipTrigger>
              <TooltipContent>Stop generating</TooltipContent>
            </Tooltip>
          ) : (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button type="submit" size="icon" className="send-button" disabled={!value.trim()} aria-label="Send question"><Send size={17} /></Button>
              </TooltipTrigger>
              <TooltipContent>Send message</TooltipContent>
            </Tooltip>
          )}
        </div>
      </div>
    </form>
  );
}

function ThreadList({ threads, activeId, loading, onOpen, onRename, onDelete }: {
  threads: ThreadSummary[];
  activeId: string | null;
  loading: boolean;
  onOpen: (id: string) => void;
  onRename: (thread: ThreadSummary) => void;
  onDelete: (thread: ThreadSummary) => void;
}) {
  return (
    <ScrollArea className="thread-scroll">
      <div className="thread-list">
        <span className="thread-label">Recent conversations</span>
        {loading && (
          <div className="thread-skeletons" aria-label="Loading conversations">
            <Skeleton /><Skeleton /><Skeleton />
          </div>
        )}
        {!loading && threads.length === 0 && (
          <div className="thread-empty"><MessageSquare size={18} /><p>No saved conversations yet.</p></div>
        )}
        {threads.map((thread) => (
          <div className={`thread-item ${thread.id === activeId ? "active" : ""}`} key={thread.id}>
            <button className="thread-open" onClick={() => onOpen(thread.id)} title={thread.title} aria-current={thread.id === activeId ? "page" : undefined}>
              <MessageSquare size={15} />
              <span>{thread.title}</span>
            </button>
            <div className="thread-actions">
              <Button variant="ghost" size="icon" onClick={() => onRename(thread)} aria-label={`Rename ${thread.title}`} title="Rename conversation"><Pencil size={14} /></Button>
              <Button variant="ghost" size="icon" className="thread-delete" onClick={() => onDelete(thread)} aria-label={`Delete ${thread.title}`} title="Delete conversation"><Trash2 size={14} /></Button>
            </div>
          </div>
        ))}
      </div>
    </ScrollArea>
  );
}

function ThreadSidebar({ threads, activeId, loading, onOpen, onRename, onDelete, mobile = false }: {
  threads: ThreadSummary[];
  activeId: string | null;
  loading: boolean;
  onOpen: (id: string) => void;
  onRename: (thread: ThreadSummary) => void;
  onDelete: (thread: ThreadSummary) => void;
  mobile?: boolean;
}) {
  const content = (
    <>
      <div className="sidebar-heading">
        {mobile && <Brand />}
      </div>
      <ThreadList {...{ threads, activeId, loading, onOpen, onRename, onDelete }} />
    </>
  );
  return mobile ? <div className="mobile-sidebar">{content}</div> : <aside className="thread-sidebar">{content}</aside>;
}

function MarkdownContent({ text, response, onSource }: { text: string; response?: AnswerResponse; onSource?: (index: number) => void }) {
  const renderInline = (value: string, keyPrefix: string) => value.split(/(\*\*[^*]+\*\*|\[E\d+\])/g).map((part, index) => {
    const label = /^\[(E\d+)\]$/.exec(part)?.[1];
    const citationIndex = label && response ? response.citations.findIndex((item) => item.label === label) : -1;
    if (citationIndex >= 0 && onSource) {
      return <button key={`${keyPrefix}-${index}`} className="citation" onClick={() => onSource(citationIndex)} title={`Open source ${label}`}>{label}</button>;
    }
    if (/^\*\*[^*]+\*\*$/.test(part)) return <strong key={`${keyPrefix}-${index}`}>{part.slice(2, -2)}</strong>;
    return <span key={`${keyPrefix}-${index}`}>{part}</span>;
  });

  const splitRow = (line: string) => {
    const escapedPipe = "\u0000";
    const normalized = line.trim().replace(/\\\|/g, escapedPipe).replace(/^\||\|$/g, "");
    return normalized.split("|").map((cell) => cell.trim().split(escapedPipe).join("|"));
  };
  const lines = text.split(/\r?\n/);
  const blocks: Array<{ type: "text"; value: string } | { type: "table"; rows: string[][] }> = [];
  let textLines: string[] = [];
  const flushText = () => {
    if (textLines.length) blocks.push({ type: "text", value: textLines.join("\n") });
    textLines = [];
  };
  for (let index = 0; index < lines.length;) {
    const header = splitRow(lines[index]);
    const separator = index + 1 < lines.length ? splitRow(lines[index + 1]) : [];
    const isTable = header.length > 1 && separator.length === header.length
      && separator.every((cell) => /^:?-{3,}:?$/.test(cell));
    if (!isTable) {
      textLines.push(lines[index]);
      index += 1;
      continue;
    }
    flushText();
    const rows = [header];
    index += 2;
    while (index < lines.length && lines[index].includes("|")) {
      const row = splitRow(lines[index]);
      if (row.length !== header.length) break;
      rows.push(row);
      index += 1;
    }
    blocks.push({ type: "table", rows });
  }
  flushText();

  return (
    <div className="answer-text">
      {blocks.map((block, blockIndex) => block.type === "text" ? (
        <div className="answer-prose" key={`text-${blockIndex}`}>{renderInline(block.value, `text-${blockIndex}`)}</div>
      ) : (
        <div className="answer-table-scroll" key={`table-${blockIndex}`}>
          <table>
            <thead><tr>{block.rows[0].map((cell, cellIndex) => <th key={cellIndex}>{renderInline(cell, `th-${blockIndex}-${cellIndex}`)}</th>)}</tr></thead>
            <tbody>{block.rows.slice(1).map((row, rowIndex) => (
              <tr key={rowIndex}>{row.map((cell, cellIndex) => <td key={cellIndex}>{renderInline(cell, `td-${blockIndex}-${rowIndex}-${cellIndex}`)}</td>)}</tr>
            ))}</tbody>
          </table>
        </div>
      ))}
    </div>
  );
}

function AnswerText({ response, onSource }: { response: AnswerResponse; onSource: (index: number) => void }) {
  return <MarkdownContent text={response.answer} response={response} onSource={onSource} />;
}

function AssistantTurn({ turn, copied, onCopy, onSource }: {
  turn: Turn;
  copied: boolean;
  onCopy: () => void;
  onSource: (response: AnswerResponse, index: number) => void;
}) {
  return (
    <article className="assistant-turn">
      <div className="assistant-heading">
        <span className="assistant-mark"><Sparkles size={15} /></span>
        <div>
          <strong>BankScope</strong>
          <small>{turn.state === "loading" ? turn.status || "Preparing the answer…" : "Grounded in indexed filings"}</small>
        </div>
      </div>
      {turn.state === "loading" && (
        <div className="thinking-card" role="status">
          <span className="spinner" />
          <div><Skeleton /><Skeleton /><Skeleton /></div>
        </div>
      )}
      {turn.state === "error" && (
        <div className="error-card" role="alert"><strong>Answer could not be generated.</strong><p>{turn.error}</p></div>
      )}
      {turn.state === "answered" && turn.response && (
        <div className="answer-body">
          <AnswerText response={turn.response} onSource={(index) => onSource(turn.response!, index)} />
          <div className="answer-meta">
            {turn.response.ticker && <Badge variant="secondary">{turn.response.ticker} detected</Badge>}
            <span>{turn.response.citations.length} {turn.response.citations.length === 1 ? "source" : "sources"}</span>
            <span className="meta-separator" />
            <span>{turn.response.status}</span>
          </div>
          <div className="answer-actions">
            <Button variant="ghost" size="sm" onClick={onCopy}><Copy size={14} /> {copied ? "Copied" : "Copy"}</Button>
            {turn.response.citations.length > 0 && (
              <Button variant="ghost" size="sm" onClick={() => onSource(turn.response!, 0)}><FileSearch size={14} /> Sources</Button>
            )}
          </div>
        </div>
      )}
    </article>
  );
}

function metadataValue(chunk: CitationContext["chunks"][number] | undefined, key: string): string {
  const value = chunk?.metadata?.[key];
  return typeof value === "string" || typeof value === "number" ? String(value) : "";
}

function SourcePanel({ source, onChange, onClose }: { source: OpenSource; onChange: (index: number) => void; onClose: () => void }) {
  const citation = source.response.citations[source.index];
  const [context, setContext] = useState<CitationContext | null>(null);
  const [error, setError] = useState<{ message: string; stale: boolean } | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    loadCitationContext(citation.citation_id, controller.signal)
      .then(setContext)
      .catch((caught: unknown) => {
        if (controller.signal.aborted) return;
        const apiError = caught instanceof ApiError ? caught : null;
        setError({
          message: caught instanceof Error ? caught.message : "The citation source is unavailable.",
          stale: apiError?.code === "citation_corpus_mismatch",
        });
      });
    return () => controller.abort();
  }, [citation.citation_id]);

  const anchor = context?.chunks.find((chunk) => chunk.role === "anchor");
  const page = (citation.display_page_start ?? citation.page_start ?? metadataValue(anchor, "start_display_page")) || metadataValue(anchor, "page_start");
  const section = citation.section_title || metadataValue(anchor, "section_title") || "Filing evidence";
  const filingDate = citation.filing_date || metadataValue(anchor, "filing_date");

  return (
    <Sheet open onOpenChange={(open) => { if (!open) onClose(); }}>
      <SheetContent className="source-panel" aria-describedby="source-description">
        <div className="source-header">
          <div>
            <span>Evidence viewer</span>
            <SheetTitle>{source.response.citations.length} {source.response.citations.length === 1 ? "source" : "sources"}</SheetTitle>
            <SheetDescription id="source-description">Canonical filing context for this answer.</SheetDescription>
          </div>
        </div>
        <div className="source-tabs" role="tablist" aria-label="Answer sources">
          {source.response.citations.map((item, index) => (
            <button key={item.citation_id} role="tab" aria-selected={index === source.index} className={index === source.index ? "active" : ""} onClick={() => onChange(index)}>{item.label}</button>
          ))}
        </div>
        <ScrollArea className="source-scroll">
          <div className="source-card">
            <div className="source-title">
              <Badge>{citation.ticker || source.response.ticker || "SEC"}</Badge>
              <div><strong>{section}</strong><small>{[citation.record_type, page ? `page ${page}` : "", filingDate].filter(Boolean).join(" · ")}</small></div>
            </div>
            {!context && !error && <div className="source-loading"><span className="spinner" /> Loading canonical evidence…</div>}
            {error && <div className={`source-error ${error.stale ? "stale" : ""}`}><strong>{error.stale ? "Source version changed" : "Source unavailable"}</strong><p>{error.message}</p></div>}
            {context?.chunks.map((chunk) => (
              <section className={`context-chunk ${chunk.role}`} key={`${chunk.target_chunk_id}-${chunk.role}`}>
                <small>{chunk.role === "anchor" ? "Cited evidence" : `${chunk.role} context`}</small>
                <MarkdownContent text={chunk.document} />
              </section>
            ))}
            {(context?.source_url || citation.source_url) && (
              <Button asChild variant="outline" size="sm"><a href={context?.source_url || citation.source_url} target="_blank" rel="noreferrer">Open original filing <ArrowRight size={14} /></a></Button>
            )}
          </div>
        </ScrollArea>
        <div className="source-navigation">
          <Button variant="ghost" size="sm" disabled={source.index === 0} onClick={() => onChange(source.index - 1)}><ChevronLeft size={15} /> Previous</Button>
          <span>{source.index + 1} of {source.response.citations.length}</span>
          <Button variant="ghost" size="sm" disabled={source.index === source.response.citations.length - 1} onClick={() => onChange(source.index + 1)}>Next <ChevronRight size={15} /></Button>
        </div>
      </SheetContent>
    </Sheet>
  );
}

export default function App() {
  const navigate = useNavigate();
  const match = useMatch("/chats/:threadId");
  const activeThreadId = match?.params.threadId ?? null;
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [threadsLoading, setThreadsLoading] = useState(true);
  const [loadedThreadId, setLoadedThreadId] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [openSource, setOpenSource] = useState<OpenSource | null>(null);
  const [pageError, setPageError] = useState<string | null>(null);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [renameTarget, setRenameTarget] = useState<ThreadSummary | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<ThreadSummary | null>(null);
  const [showScrollDown, setShowScrollDown] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const conversationRef = useRef<HTMLDivElement>(null);
  const activeRequestRef = useRef<AbortController | null>(null);
  const shouldAutoScrollRef = useRef(true);

  const refreshThreads = async () => {
    const next = await listThreads();
    setThreads(next);
    return next;
  };

  useEffect(() => {
    const controller = new AbortController();
    listThreads(controller.signal)
      .then(setThreads)
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setPageError(error instanceof Error ? error.message : "Could not load conversations.");
      })
      .finally(() => { if (!controller.signal.aborted) setThreadsLoading(false); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!activeThreadId || loadedThreadId === activeThreadId) return;
    const controller = new AbortController();
    loadThread(activeThreadId, controller.signal)
      .then((history) => {
        setTurns(history.turns);
        setLoadedThreadId(activeThreadId);
        shouldAutoScrollRef.current = true;
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setPageError(error instanceof Error ? error.message : "Could not load conversation.");
      });
    return () => controller.abort();
  }, [activeThreadId, loadedThreadId]);

  useEffect(() => {
    if (shouldAutoScrollRef.current && typeof endRef.current?.scrollIntoView === "function") {
      endRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [turns]);

  const ask = async (suggestedQuestion?: string) => {
    const nextQuestion = (suggestedQuestion ?? question).trim();
    if (!nextQuestion || loading) return;
    let threadId = activeThreadId;
    try {
      if (!threadId) {
        const thread = await createThread();
        threadId = thread.id;
        setThreads((current) => [thread, ...current]);
        setLoadedThreadId(thread.id);
        navigate(`/chats/${thread.id}`);
      }
      const optimisticId = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}`;
      const controller = new AbortController();
      activeRequestRef.current = controller;
      shouldAutoScrollRef.current = true;
      setShowScrollDown(false);
      setQuestion("");
      setLoading(true);
      setTurns((current) => [...current, { id: optimisticId, question: nextQuestion, state: "loading", status: stageLabels.resolving_bank }]);
      const turn = await streamAnswer(
        threadId,
        nextQuestion,
        (stage, message) => setTurns((current) => current.map((item) => item.id === optimisticId ? { ...item, status: message || stageLabels[stage] } : item)),
        controller.signal,
      );
      setTurns((current) => current.map((item) => item.id === optimisticId ? turn : item));
      await refreshThreads();
    } catch (error) {
      if (activeRequestRef.current?.signal.aborted) {
        setTurns((current) => current.filter((turn) => turn.state !== "loading"));
        return;
      }
      const message = error instanceof Error ? error.message : "Unexpected error while contacting the answer service.";
      setTurns((current) => current.map((turn) => turn.state === "loading" ? { ...turn, state: "error", error: message } : turn));
    } finally {
      activeRequestRef.current = null;
      setLoading(false);
    }
  };

  const stopGenerating = () => activeRequestRef.current?.abort();

  const newConversation = () => {
    activeRequestRef.current?.abort();
    setQuestion("");
    setTurns([]);
    setLoadedThreadId(null);
    setOpenSource(null);
    setMobileNavOpen(false);
    navigate("/");
  };

  const openThread = (id: string) => {
    setPageError(null);
    setMobileNavOpen(false);
    shouldAutoScrollRef.current = true;
    navigate(`/chats/${id}`);
  };

  const beginRename = (thread: ThreadSummary) => {
    setMobileNavOpen(false);
    setRenameTarget(thread);
    setRenameValue(thread.title);
  };

  const beginDelete = (thread: ThreadSummary) => {
    setMobileNavOpen(false);
    setDeleteTarget(thread);
  };

  const handleRename = async (event: FormEvent) => {
    event.preventDefault();
    const title = renameValue.trim();
    if (!renameTarget || !title) return;
    try {
      const updated = await renameThread(renameTarget.id, title);
      setThreads((current) => current.map((item) => item.id === updated.id ? updated : item));
      setRenameTarget(null);
    } catch (error) { setPageError(error instanceof Error ? error.message : "Could not rename conversation."); }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      await deleteThread(deleteTarget.id);
      setThreads((current) => current.filter((item) => item.id !== deleteTarget.id));
      if (activeThreadId === deleteTarget.id) newConversation();
      setDeleteTarget(null);
    } catch (error) { setPageError(error instanceof Error ? error.message : "Could not delete conversation."); }
  };

  const copyAnswer = async (turn: Turn) => {
    if (!turn.response) return;
    await navigator.clipboard.writeText(turn.response.answer);
    setCopiedId(turn.id);
    window.setTimeout(() => setCopiedId(null), 1400);
  };

  const onConversationScroll = () => {
    const element = conversationRef.current;
    if (!element) return;
    const nearBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 100;
    shouldAutoScrollRef.current = nearBottom;
    setShowScrollDown(!nearBottom);
  };

  const scrollToBottom = () => {
    shouldAutoScrollRef.current = true;
    setShowScrollDown(false);
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  };

  const historyLoading = Boolean(activeThreadId && loadedThreadId !== activeThreadId);
  const showWelcome = !activeThreadId || (!historyLoading && turns.length === 0);
  const sidebarProps = { threads, activeId: activeThreadId, loading: threadsLoading, onOpen: openThread, onRename: beginRename, onDelete: beginDelete };

  return (
    <TooltipProvider delayDuration={300}>
      <div className="app-shell persistent-shell">
        <header className="app-header">
          <div className="header-left">
            <Button variant="ghost" size="icon" className="mobile-menu-button" onClick={() => setMobileNavOpen(true)} aria-label="Open conversations"><Menu size={20} /></Button>
            <Brand />
          </div>
          <div className="header-actions">
            <Badge variant="success" className="corpus-status"><i /> Corpus ready</Badge>
            <Button variant="secondary" className="new-chat" onClick={newConversation}><Plus size={16} /> <span>New conversation</span></Button>
          </div>
        </header>
        <div className="workspace">
          <ThreadSidebar {...sidebarProps} />
          <div className="workspace-main">
            {pageError && <div className="page-error" role="alert"><span>{pageError}</span><Button variant="ghost" size="sm" onClick={() => setPageError(null)}>Dismiss</Button></div>}
            {historyLoading ? (
              <main className="welcome"><div className="source-loading"><span className="spinner" /> Loading conversation…</div></main>
            ) : showWelcome ? (
              <main className="welcome">
                <section className="welcome-content">
                  <div className="eyebrow"><Sparkles size={14} /> BankScope</div>
                  <h1>Ask any question.<br /><span>Follow the evidence.</span></h1>
                  <p>Ask questions across the latest indexed 10-K filings from 10 leading U.S. banks. Every answer is grounded in verifiable filing evidence.</p>
                  <Composer value={question} onChange={setQuestion} onSubmit={ask} onStop={stopGenerating} loading={loading} />
                  <div className="suggestions"><span>Try a research prompt</span><div>
                    {prompts.map((prompt) => (
                      <button key={prompt.label} onClick={() => void ask(prompt.text)}>
                        <span><strong>{prompt.label}</strong><small>{prompt.text}</small></span><ArrowRight size={16} />
                      </button>
                    ))}
                  </div></div>
                  <div className="trust-line"><BadgeCheck size={15} /> Answers use only indexed filing evidence.</div>
                </section>
              </main>
            ) : (
              <main className="conversation">
                <div className="conversation-scroll" ref={conversationRef} onScroll={onConversationScroll}>
                  <div className="conversation-list">
                    {turns.map((turn) => (
                      <section className="turn" key={turn.id}>
                        <div className="user-turn"><span>You</span><p>{turn.question}</p></div>
                        <AssistantTurn turn={turn} copied={copiedId === turn.id} onCopy={() => void copyAnswer(turn)} onSource={(response, index) => setOpenSource({ response, index })} />
                      </section>
                    ))}
                    <div ref={endRef} />
                  </div>
                </div>
                {showScrollDown && <Button variant="outline" size="icon" className="scroll-bottom" onClick={scrollToBottom} aria-label="Scroll to latest message"><ArrowDown size={17} /></Button>}
                <div className="conversation-composer">
                  <Composer compact value={question} onChange={setQuestion} onSubmit={ask} onStop={stopGenerating} loading={loading} />
                  <small className="composer-disclaimer">BankScope can make mistakes. Verify important details in the cited filing.</small>
                </div>
              </main>
            )}
          </div>
        </div>

        <Sheet open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
          <SheetContent side="left" className="mobile-nav-sheet">
            <SheetTitle className="sr-only">Conversations</SheetTitle>
            <SheetDescription className="sr-only">Browse and manage saved conversations.</SheetDescription>
            <ThreadSidebar {...sidebarProps} mobile />
          </SheetContent>
        </Sheet>

        <Dialog open={Boolean(renameTarget)} onOpenChange={(open) => { if (!open) setRenameTarget(null); }}>
          <DialogContent>
            <form onSubmit={(event) => void handleRename(event)}>
              <DialogHeader>
                <DialogTitle>Rename conversation</DialogTitle>
                <DialogDescription>Use a short title that makes this research easy to find later.</DialogDescription>
              </DialogHeader>
              <input className="dialog-input" value={renameValue} onChange={(event) => setRenameValue(event.target.value)} aria-label="Conversation title" autoFocus />
              <DialogFooter>
                <Button type="button" variant="outline" onClick={() => setRenameTarget(null)}>Cancel</Button>
                <Button type="submit" disabled={!renameValue.trim()}>Save title</Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>

        <AlertDialog open={Boolean(deleteTarget)} onOpenChange={(open) => { if (!open) setDeleteTarget(null); }}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete this conversation?</AlertDialogTitle>
              <AlertDialogDescription>“{deleteTarget?.title}” and its message history will be permanently removed.</AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction onClick={() => void handleDelete()}>Delete conversation</AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        {openSource && (
          <SourcePanel key={openSource.response.citations[openSource.index].citation_id} source={openSource} onChange={(index) => setOpenSource({ ...openSource, index })} onClose={() => setOpenSource(null)} />
        )}
      </div>
    </TooltipProvider>
  );
}
