import { lazy, Suspense, type FormEvent, type KeyboardEvent, useEffect, useId, useRef, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { toast } from "sonner";
import {
  ArrowDown,
  ArrowRight,
  BadgeCheck,
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
  createThread,
  deleteThread,
  listThreads,
  loadThread,
  renameThread,
  streamAnswer,
  type AnswerResponse,
  type Diagnostics,
  type EvidenceAudit,
  type ThreadSummary,
  type Turn,
} from "./api";
import { prompts } from "./data";
import { uploadDocument } from "./api";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { MotionButton } from "@/components/ui/motion-button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/motion-dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/motion-sheet";
import { Skeleton } from "@/components/ui/motion-skeleton";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { FileUploadButton } from "@/components/ui/file-upload-button";
import { FileList } from "@/components/ui/file-list";
import { container, item, spring } from "@/components/ui/motion-presets";
import { FeatureErrorBoundary } from "@/components/FeatureErrorBoundary";
import type { OpenSource } from "@/features/citations/SourcePanel";

const SourcePanel = lazy(() => import("@/features/citations/SourcePanel"));
type AnswerStatus = AnswerResponse["status"];

const statusVariants: Record<AnswerStatus, "success" | "warning" | "danger"> = {
  supported: "success",
  partial: "warning",
  ambiguous: "warning",
  unsupported: "danger",
};

const stageLabels: Record<string, string> = {
  routing: "Routing the request\u2026",
  assessing_evidence: "Assessing retrieved evidence\u2026",
  auditing_evidence: "Reviewing the answer against cited evidence\u2026",
  rewriting_search: "Running a refined filing search\u2026",
  expanding_context: "Reading adjacent filing context\u2026",
  resolving_bank: "Identifying the bank…",
  embedding: "Encoding the question…",
  retrieving: "Searching indexed filings…",
  generating: "Generating a grounded answer…",
  synthesizing: "Synthesizing the bank comparison…",
  validating: "Validating citations and answer structure…",
};

function DiagnosticsPanel({ diagnostics }: { diagnostics?: Diagnostics }) {
  const [expanded, setExpanded] = useState(false);
  const diagnosticsId = useId();
  if (!diagnostics) return null;
  const qualityGate = diagnostics.quality_gate ?? { passed: false, checks: {} };
  const checks = Object.entries(qualityGate.checks ?? {});
  const stages = Array.isArray(diagnostics.stages) ? diagnostics.stages : [];
  const bankPlans = Array.isArray(diagnostics.bank_plans) ? diagnostics.bank_plans : [];
  return (
    <>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        className="diagnostics-trigger"
        aria-expanded={expanded}
        aria-controls={diagnosticsId}
        onClick={() => setExpanded((current) => !current)}
      >
        <ChevronRight size={14} className={expanded ? "expanded" : ""} /> Diagnostics
      </Button>
      {expanded && <div className="diagnostics-content" id={diagnosticsId} role="region" aria-label="Diagnostics details">
        <div className="diagnostics-summary">
          <span>Route: <strong>{diagnostics.route}</strong></span>
          <span>Agentic RAG: <strong>{diagnostics.agentic_rag_enabled ? "enabled" : "disabled"}</strong></span>
          <span>Evidence: <strong>{diagnostics.initial_evidence_count ?? "\u2014"} {"\u2192"} {diagnostics.final_evidence_count ?? "\u2014"}</strong></span>
          <span>Model requests: <strong>{diagnostics.model_request_count ?? "\u2014"}</strong></span>
        </div>
        {stages.length > 0 && (
          <ol className="diagnostics-timeline">
            {stages.map((stage, index) => (
              <li key={`${stage.stage}-${index}`}>
                <span>{stage.stage.replace(/_/g, " ")}</span>
                {typeof stage.latency_ms === "number" && <small>{stage.latency_ms.toFixed(1)} ms</small>}
              </li>
            ))}
          </ol>
        )}
        {bankPlans.map((plan) => (
          <div className="diagnostics-plan" key={plan.ticker}>
            <strong>{plan.ticker}: {plan.action}{plan.final_status ? ` \u2192 ${plan.final_status}` : ""}</strong>
            {plan.reason_code && <span>{plan.reason_code}</span>}
            {plan.explanation && <p>{plan.explanation}</p>}
            {typeof plan.model_request_count === "number" && (
              <span>{plan.model_request_count} model / {plan.tool_action_count ?? 0} tool / {plan.verifier_request_count ?? 0} verifier</span>
            )}
            {plan.rewritten_query && <code>{plan.rewritten_query}</code>}
            {plan.anchor_target_chunk_id && <code>{plan.anchor_target_chunk_id}</code>}
            {plan.steps && plan.steps.length > 0 && (
              <ol>
                {plan.steps.map((step, index) => (
                  <li key={`${step.action}-${index}`}>
                    {step.action.replace(/_/g, " ")}
                    {step.query ? `: ${step.query}` : ""}
                    {step.terms ? `: ${step.terms.join(", ")}` : ""}
                    {step.result ? ` - ${step.result}` : ""}
                    {step.error_code ? ` - ${step.error_code}` : ""}
                  </li>
                ))}
              </ol>
            )}
          </div>
        ))}
        <div className={`execution-checks ${qualityGate.passed ? "passed" : "failed"}`}>
          <strong>Execution checks: {qualityGate.passed ? "passed" : "failed"}</strong>
          <ul>{checks.map(([name, passed]) => (
            <li key={name}>
              <span className={`diagnostics-check-icon ${passed ? "passed" : "failed"}`} aria-hidden="true">{passed ? "\u2713" : "\u2715"}</span>
              {name.replace(/_/g, " ")}
            </li>
          ))}</ul>
        </div>
      </div>}
    </>
  );
}

function EvidenceAuditPanel({ audit }: { audit?: EvidenceAudit }) {
  const [expanded, setExpanded] = useState(false);
  const auditId = useId();
  if (!audit) return null;
  const label = audit.status === "passed"
    ? "Passed"
    : audit.status === "review_recommended"
    ? "Review recommended"
    : "Unavailable";
  const unavailable = audit.status === "unavailable";
  const checks = [
    ["Question addressed", audit.question_addressed],
    ["Material claims grounded", audit.grounded],
    ["Citation coverage", audit.citation_coverage_ok],
    ["Contradiction found", audit.contradiction_found],
  ] as const;
  return (
    <>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        className={`evidence-audit-trigger evidence-audit-${audit.status}`}
        aria-expanded={expanded}
        aria-controls={auditId}
        onClick={() => setExpanded((current) => !current)}
      >
        <ChevronRight size={14} className={expanded ? "expanded" : ""} /> Evidence audit: {label}
      </Button>
      {expanded && (
        <div className="evidence-audit-content" id={auditId} role="region" aria-label="Evidence audit details">
          <ul>
            {checks.map(([name, value]) => (
              <li key={name}>
                <span>{name}</span>
                <strong>{unavailable ? "Not assessed" : value ? "Yes" : "No"}</strong>
              </li>
            ))}
          </ul>
          <p>{audit.summary}</p>
          <small>Automated review against the cited evidence; not a guarantee of correctness.</small>
        </div>
      )}
    </>
  );
}

function Brand() {
  return (
    <div className="brand" aria-label="BankScope home">
      <img className="brand-wordmark" src="/brand/bankscope-wordmark.svg" alt="BankScope" />
    </div>
  );
}

function StatusBadge({ status }: { status: AnswerStatus }) {
  return <Badge variant={statusVariants[status]} data-status={status}>{status}</Badge>;
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
        placeholder="Ask naturally, research a filing, search the web, or calculate…"
        rows={compact ? 1 : 3}
        aria-label="Research question"
        disabled={loading}
      />
      <div className="composer-footer">
        <span><Sparkles size={13} /> Bank and filing detection is automatic</span>
        <motion.div className="send-group" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
          {!compact && <small>Enter to send · Shift + Enter for a new line</small>}
          {loading ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <MotionButton type="button" size="icon" className="send-button stop-button" whileHover={{ scale: 1 }} onClick={onStop} aria-label="Stop generating"><Square size={14} fill="currentColor" /></MotionButton>
              </TooltipTrigger>
              <TooltipContent>Stop generating</TooltipContent>
            </Tooltip>
          ) : (
            <Tooltip>
              <TooltipTrigger asChild>
                <MotionButton type="submit" size="icon" className="send-button" whileHover={{ scale: 1 }} disabled={!value.trim()} aria-label="Send question"><Send size={17} /></MotionButton>
              </TooltipTrigger>
              <TooltipContent>Send message</TooltipContent>
            </Tooltip>
          )}
        </motion.div>
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
      <motion.div
        className="thread-list"
        variants={container}
        initial="hidden"
        animate="visible"
      >
        <span className="thread-label">Recent conversations</span>
        {loading && (
          <div className="thread-skeletons" aria-label="Loading conversations">
            <Skeleton className="thread-list-skeleton" /><Skeleton className="thread-list-skeleton" /><Skeleton className="thread-list-skeleton" />
          </div>
        )}
        {!loading && threads.length === 0 && (
          <motion.div className="thread-empty" variants={item}><MessageSquare size={18} /><p>No saved conversations yet.</p></motion.div>
        )}
        {threads.map((thread) => (
          <motion.div
            key={thread.id}
            className={`thread-item ${thread.id === activeId ? "active" : ""}`}
            variants={item}
            whileHover={{ scale: 1.01 }}
            transition={spring}
          >
            <button className="thread-open" onClick={() => onOpen(thread.id)} title={thread.title} aria-current={thread.id === activeId ? "page" : undefined}>
              <MessageSquare size={15} />
              <span>{thread.title}</span>
            </button>
            <div className="thread-actions">
              <MotionButton variant="ghost" size="icon" onClick={() => onRename(thread)} aria-label={`Rename ${thread.title}`} title="Rename conversation"><Pencil size={14} /></MotionButton>
              <MotionButton variant="ghost" size="icon" className="thread-delete" onClick={() => onDelete(thread)} aria-label={`Delete ${thread.title}`} title="Delete conversation"><Trash2 size={14} /></MotionButton>
            </div>
          </motion.div>
        ))}
      </motion.div>
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
    const citation = citationIndex >= 0 ? response?.citations[citationIndex] : undefined;
    if (citation?.kind === "web") {
      return (
        <a
          key={`${keyPrefix}-${index}`}
          className="citation"
          href={citation.source_url}
          target="_blank"
          rel="noopener noreferrer"
          title={citation.title ? `Open source ${label}: ${citation.title}` : `Open source ${label}`}
        >
          {label}
        </a>
      );
    }
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
        <div className="answer-table-scroll" key={`table-${blockIndex}`} role="region" aria-label="Scrollable data table" tabIndex={0}>
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

function ComparisonResults({ response, onSource }: {
  response: AnswerResponse;
  onSource: (index: number) => void;
}) {
  if (response.mode !== "comparison" || !response.bank_results?.length) return null;
  return (
    <div className="comparison-results" aria-label="Bank comparison details">
      {response.bank_results.map((result) => {
        const sourceIndexes = result.citations
          .map((citation) => response.citations.findIndex((item) => item.label === citation.label))
          .filter((index) => index >= 0);
        return (
          <section className={`comparison-card comparison-card-${result.status}`} key={result.ticker}>
            <div className="comparison-card-heading">
              <div><Badge>{result.ticker}</Badge><strong>{result.bank_name}</strong></div>
              <StatusBadge status={result.status} />
            </div>
            <MarkdownContent text={result.answer} response={response} onSource={onSource} />
            {sourceIndexes.length > 0 && (
              <Button variant="ghost" size="sm" onClick={() => onSource(sourceIndexes[0])}>
                <FileSearch size={14} /> {sourceIndexes.length} {sourceIndexes.length === 1 ? "source" : "sources"}
              </Button>
            )}
          </section>
        );
      })}
    </div>
  );
}

function assistantSubtitle(turn: Turn): string {
  if (turn.state === "loading") return turn.status || "Preparing the answer…";
  if (turn.state !== "answered" || !turn.response) return "BankScope assistant";
  const hasDocument = turn.response.citations.some((citation) => citation.kind === "document");
  const hasFiling = turn.response.citations.some((citation) => citation.kind === "filing");
  if (
    turn.response.source_scope === "uploaded_document_and_indexed_filing"
    || turn.response.diagnostics?.route === "document_filing_comparison"
    || (hasDocument && hasFiling)
  ) return "Grounded in uploaded document and indexed filing";
  switch (turn.response.dialog_act) {
    case "clarification": return "One detail is needed";
    case "retryable_error": return "Research paused safely";
    case "contextual_transform": return "Continued from the previous answer";
    case "web_answer": return "Researched on the web";
    case "web_research_unavailable": return "Web research unavailable";
    case "calculation": return "Calculated result";
    case "answer": return hasDocument
      ? "Grounded in uploaded document"
      : "Grounded in indexed filings";
    default:
      return turn.response.citations.some((citation) => citation.kind === "web")
        ? "Researched on the web"
        : turn.response.citations.length > 0
        ? "Grounded in indexed filings"
        : "BankScope assistant";
  }
}

function hasAnswerMetadata(response: AnswerResponse): boolean {
  return Boolean(
    response.ticker
    || (response.tickers && response.tickers.length > 1)
    || response.citations.length > 0
    || response.dialog_act === "answer"
    || response.dialog_act === "clarification"
    || response.dialog_act === "retryable_error"
    || response.dialog_act === "contextual_transform"
    || response.dialog_act === "web_answer"
    || response.dialog_act === "web_research_unavailable"
    || response.dialog_act === "calculation"
    || response.mode === "comparison",
  );
}

function AssistantTurn({ turn, copied, onCopy, onSource, onRetry, retryDisabled }: {
  turn: Turn;
  copied: boolean;
  onCopy: () => void;
  onSource: (response: AnswerResponse, index: number) => void;
  onRetry: (question: string) => void;
  retryDisabled: boolean;
}) {
  return (
    <motion.article
      className="assistant-turn"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      <div className="assistant-heading">
        <span className="assistant-mark" aria-hidden="true"><img src="/brand/bankscope-target.svg" alt="" /></span>
        <div>
          <strong>BankScope</strong>
          <small>{assistantSubtitle(turn)}</small>
        </div>
      </div>
      {turn.state === "loading" && (
        <motion.div
          className="thinking-card"
          role="status"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={spring}
        >
          <motion.span className="spinner" animate={{ rotate: 360 }} transition={{ duration: 0.7, repeat: Infinity, ease: "linear" }} />
          <div><Skeleton className="answer-skeleton answer-skeleton-wide" /><Skeleton className="answer-skeleton answer-skeleton-short" /><Skeleton className="answer-skeleton answer-skeleton-medium" /></div>
        </motion.div>
      )}
      {turn.state === "error" && <div className="answer-actions error-diagnostics"><DiagnosticsPanel diagnostics={turn.diagnostics} /></div>}
      {turn.state === "error" && (
        <motion.div
          className="error-card"
          role="alert"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3 }}
        >
          <strong>Answer could not be generated.</strong><p>{turn.error}</p>
        </motion.div>
      )}
      {turn.state === "answered" && turn.response && (
        <motion.div
          className="answer-body"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
        >
          <AnswerText response={turn.response} onSource={(index) => onSource(turn.response!, index)} />
          <ComparisonResults response={turn.response} onSource={(index) => onSource(turn.response!, index)} />
          {hasAnswerMetadata(turn.response) && (
            <motion.div
              className="answer-meta"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
            >
              {turn.response.ticker && <Badge variant="secondary">{turn.response.ticker} detected</Badge>}
              {turn.response.tickers && turn.response.tickers.length > 1 && (
                <Badge variant="secondary">{turn.response.tickers.join(" vs ")}</Badge>
              )}
              {turn.response.citations.length > 0 && (
                <span>{turn.response.citations.length} {turn.response.citations.length === 1 ? "source" : "sources"}</span>
              )}
              {turn.response.dialog_act === "clarification" && <Badge variant="secondary">Clarification</Badge>}
              {turn.response.dialog_act === "retryable_error" && <Badge variant="secondary">Try again</Badge>}
              {turn.response.dialog_act === "contextual_transform" && <Badge variant="secondary">Follow-up</Badge>}
              {turn.response.dialog_act === "web_answer" && <Badge variant="secondary">Web research</Badge>}
              {turn.response.dialog_act === "web_research_unavailable" && <Badge variant="secondary">Web unavailable</Badge>}
              {turn.response.dialog_act === "calculation" && <Badge variant="secondary">Calculation</Badge>}
              {(turn.response.dialog_act === "answer" || turn.response.citations.length > 0 || turn.response.mode === "comparison") && (
                <><span className="meta-separator" /><StatusBadge status={turn.response.status} /></>
              )}
            </motion.div>
          )}
          <motion.div
            className="answer-actions"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <MotionButton variant="ghost" size="sm" onClick={onCopy}><Copy size={14} /> {copied ? "Copied" : "Copy"}</MotionButton>
            {turn.response.dialog_act === "retryable_error" && (
              <MotionButton variant="outline" size="sm" disabled={retryDisabled} onClick={() => onRetry(turn.question)}>Retry</MotionButton>
            )}
            {turn.response.citations.length > 0 && (
              <MotionButton variant="ghost" size="sm" onClick={() => onSource(turn.response!, 0)}><FileSearch size={14} /> Sources</MotionButton>
            )}
            <EvidenceAuditPanel audit={turn.response.evidence_audit} />
            <DiagnosticsPanel diagnostics={turn.response.diagnostics} />
          </motion.div>
        </motion.div>
      )}
    </motion.article>
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
  const [fileRefreshKey, setFileRefreshKey] = useState(0);
  const endRef = useRef<HTMLDivElement>(null);
  const conversationRef = useRef<HTMLDivElement>(null);
  const activeRequestRef = useRef<AbortController | null>(null);
  const shouldAutoScrollRef = useRef(true);
  const locallyCreatedThreadIdsRef = useRef(new Set<string>());
  const initialThreadListPendingRef = useRef(true);

  const refreshThreads = async () => {
    const next = await listThreads();
    setThreads(next);
    return next;
  };

  const refreshFiles = () => {
    setFileRefreshKey((prev) => prev + 1);
  };

  const handleUploadDocument = async (file: File, threadId?: string) => {
    let targetThreadId = threadId;
    let newThread: ThreadSummary | null = null;

    if (!targetThreadId) {
      newThread = await createThread(`File: ${file.name}`);
      targetThreadId = newThread.id;
    }

    try {
      await uploadDocument(file, targetThreadId);
    } catch (error) {
      if (newThread) {
        try {
          await deleteThread(newThread.id);
        } catch (cleanupError) {
          setPageError(cleanupError instanceof Error ? cleanupError.message : "Could not remove the empty conversation.");
        }
      }
      throw error;
    }

    if (newThread) {
      if (initialThreadListPendingRef.current) locallyCreatedThreadIdsRef.current.add(newThread.id);
      setQuestion("");
      setTurns([]);
      setOpenSource(null);
      setThreads((current) => [newThread, ...current]);
      setLoadedThreadId(newThread.id);
      navigate(`/chats/${newThread.id}`);
    }
    refreshFiles();
  };

  useEffect(() => {
    const controller = new AbortController();
    listThreads(controller.signal)
      .then((listedThreads) => {
        if (controller.signal.aborted) return;
        setThreads((current) => {
          const listedIds = new Set(listedThreads.map((thread) => thread.id));
          const locallyCreated = current.filter((thread) => (
            locallyCreatedThreadIdsRef.current.has(thread.id) && !listedIds.has(thread.id)
          ));
          return [...locallyCreated, ...listedThreads];
        });
        locallyCreatedThreadIdsRef.current.clear();
        initialThreadListPendingRef.current = false;
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          initialThreadListPendingRef.current = false;
          setPageError(error instanceof Error ? error.message : "Could not load conversations.");
        }
      })
      .finally(() => { if (!controller.signal.aborted) setThreadsLoading(false); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!activeThreadId || loadedThreadId === activeThreadId) return;
    const controller = new AbortController();
    loadThread(activeThreadId, controller.signal)
      .then((history) => {
        if (controller.signal.aborted) return;
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
    if (!nextQuestion || loading || activeRequestRef.current) return;
    const controller = new AbortController();
    activeRequestRef.current = controller;
    setLoading(true);
    let threadId = activeThreadId;
    try {
      if (!threadId) {
        const thread = await createThread();
        threadId = thread.id;
        if (initialThreadListPendingRef.current) locallyCreatedThreadIdsRef.current.add(thread.id);
        setTurns([]);
        setThreads((current) => [thread, ...current]);
        setLoadedThreadId(thread.id);
        navigate(`/chats/${thread.id}`);
      }
      const optimisticId = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}`;
      shouldAutoScrollRef.current = true;
      setShowScrollDown(false);
      setQuestion("");
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
      if (controller.signal.aborted) {
        setTurns((current) => current.filter((turn) => turn.state !== "loading"));
        return;
      }
      const message = error instanceof Error ? error.message : "Unexpected error while contacting the answer service.";
      setTurns((current) => current.map((turn) => turn.state === "loading" ? { ...turn, state: "error", error: message } : turn));
    } finally {
      if (activeRequestRef.current === controller) activeRequestRef.current = null;
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
      toast.success("Conversation renamed");
    } catch (error) { setPageError(error instanceof Error ? error.message : "Could not rename conversation."); }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      await deleteThread(deleteTarget.id);
      setThreads((current) => current.filter((item) => item.id !== deleteTarget.id));
      if (activeThreadId === deleteTarget.id) newConversation();
      setDeleteTarget(null);
      toast.success("Conversation deleted");
    } catch (error) { setPageError(error instanceof Error ? error.message : "Could not delete conversation."); }
  };

  const copyAnswer = async (turn: Turn) => {
    if (!turn.response) return;
    await navigator.clipboard.writeText(turn.response.answer);
    setCopiedId(turn.id);
    toast.success("Answer copied to clipboard");
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

  const openCitation = (response: AnswerResponse, index: number) => {
    const citation = response.citations[index];
    if (!citation) return;
    if (citation.kind === "web") {
      window.open(citation.source_url, "_blank", "noopener,noreferrer");
      return;
    }
    setOpenSource({ response, index, citation });
  };

  const historyLoading = Boolean(activeThreadId && loadedThreadId !== activeThreadId);
  const showWelcome = !activeThreadId || (!historyLoading && turns.length === 0);
  const sidebarProps = { threads, activeId: activeThreadId, loading: threadsLoading, onOpen: openThread, onRename: beginRename, onDelete: beginDelete };

  return (
    <TooltipProvider delayDuration={300}>
      <div className="app-shell persistent-shell">
        <motion.header
          className="app-header"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <div className="header-left">
            <MotionButton variant="ghost" size="icon" className="mobile-menu-button" onClick={() => setMobileNavOpen(true)} aria-label="Open conversations"><Menu size={20} /></MotionButton>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
              <Brand />
            </motion.div>
          </div>
          <motion.div
            className="header-actions"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Badge variant="success" className="corpus-status"><i /> Corpus ready</Badge>
            <FileUploadButton
              onUpload={handleUploadDocument}
              threadId={activeThreadId ?? undefined}
            />
            <MotionButton variant="secondary" className="new-chat" onClick={newConversation}><Plus size={16} /> <span>New conversation</span></MotionButton>
          </motion.div>
        </motion.header>
        <motion.div
          className="workspace"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3, delay: 0.2 }}
        >
          <ThreadSidebar {...sidebarProps} />
          <motion.div
            className="workspace-main"
            initial={{ opacity: 0, scale: 0.99 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.35 }}
          >
            {pageError && (
              <motion.div className="page-error" role="alert" initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
                <span>{pageError}</span>
                <MotionButton variant="ghost" size="sm" onClick={() => setPageError(null)}>Dismiss</MotionButton>
              </motion.div>
            )}
            {historyLoading ? (
              <main className="welcome"><div className="source-loading"><span className="spinner" /> Loading conversation…</div></main>
            ) : showWelcome ? (
              <main className="welcome">
                <motion.section
                  className="welcome-content"
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                >
                  <motion.div className="eyebrow" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
                    <Sparkles size={14} /> BankScope
                  </motion.div>
                  <motion.h1
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                  >
                    Ask any question.<br /><span>Follow the evidence.</span>
                  </motion.h1>
                  <motion.p
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                  >
                    Chat naturally, calculate, research indexed bank filings, or search current web sources. Tool-based answers keep their evidence attached.
                  </motion.p>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                  >
                    <Composer value={question} onChange={setQuestion} onSubmit={ask} onStop={stopGenerating} loading={loading} />
                  </motion.div>
                  <motion.div className="suggestions" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}>
                    <span>Try a research prompt</span>
                    <div>
                      {prompts.map((prompt, index) => (
                        <motion.button
                          key={prompt.label}
                          onClick={() => void ask(prompt.text)}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.7 + index * 0.05 }}
                          whileHover={{ scale: 1.02, y: -2 }}
                          whileTap={{ scale: 0.98 }}
                        >
                          <span><strong>{prompt.label}</strong><small>{prompt.text}</small></span><ArrowRight size={16} />
                        </motion.button>
                      ))}
                    </div>
                  </motion.div>
                  <motion.div className="trust-line" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.9 }}>
                    <BadgeCheck size={15} /> Filing and web claims stay linked to verifiable sources.
                  </motion.div>
                </motion.section>
              </main>
            ) : (
              <main className="conversation">
                <div className="conversation-scroll" ref={conversationRef} onScroll={onConversationScroll}>
                  {activeThreadId && <FileList key={fileRefreshKey} threadId={activeThreadId} />}
                  <div className="conversation-list">
                    <AnimatePresence>
                      {turns.map((turn) => (
                        <motion.section
                          key={turn.id}
                          className="turn"
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -20 }}
                          transition={{ duration: 0.3, ease: "easeOut" }}
                          layout
                        >
                          <div className="user-turn"><span>You</span><p>{turn.question}</p></div>
                          <AssistantTurn
                            turn={turn}
                            copied={copiedId === turn.id}
                            onCopy={() => void copyAnswer(turn)}
                            onSource={openCitation}
                            onRetry={ask}
                            retryDisabled={loading}
                          />
                        </motion.section>
                      ))}
                    </AnimatePresence>
                    <div ref={endRef} />
                  </div>
                </div>
                {showScrollDown && <div className="scroll-bottom"><MotionButton variant="outline" size="icon" whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }} onClick={scrollToBottom} aria-label="Scroll to latest message"><ArrowDown size={17} /></MotionButton></div>}
                <motion.div
                  className="conversation-composer"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <Composer compact value={question} onChange={setQuestion} onSubmit={ask} onStop={stopGenerating} loading={loading} />
                  <small className="composer-disclaimer">BankScope can make mistakes. Verify important details in the cited sources.</small>
                </motion.div>
              </main>
            )}
          </motion.div>
        </motion.div>

        <Sheet open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
          <SheetContent open={mobileNavOpen} side="left" className="mobile-nav-sheet">
            <SheetTitle className="sr-only">Conversations</SheetTitle>
            <SheetDescription className="sr-only">Browse and manage saved conversations.</SheetDescription>
            <ThreadSidebar {...sidebarProps} mobile />
          </SheetContent>
        </Sheet>

        <Dialog open={Boolean(renameTarget)} onOpenChange={(open) => { if (!open) setRenameTarget(null); }}>
          <DialogContent open={Boolean(renameTarget)}>
            <form onSubmit={(event) => void handleRename(event)}>
              <DialogHeader>
                <DialogTitle>Rename conversation</DialogTitle>
                <DialogDescription>Use a short title that makes this research easy to find later.</DialogDescription>
              </DialogHeader>
              <input className="dialog-input" value={renameValue} onChange={(event) => setRenameValue(event.target.value)} aria-label="Conversation title" autoFocus />
              <DialogFooter>
                <MotionButton type="button" variant="outline" onClick={() => setRenameTarget(null)}>Cancel</MotionButton>
                <MotionButton type="submit" disabled={!renameValue.trim()}>Save title</MotionButton>
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

        {openSource && activeThreadId && (
          <FeatureErrorBoundary feature="Evidence viewer">
            <Suspense fallback={<div className="source-panel-loading" role="status">Opening evidence viewer…</div>}>
              <SourcePanel key={openSource.citation.citation_id} source={openSource} markdown={MarkdownContent} onChange={(index) => openCitation(openSource.response, index)} onClose={() => setOpenSource(null)} />
            </Suspense>
          </FeatureErrorBoundary>
        )}
      </div>
    </TooltipProvider>
  );
}
