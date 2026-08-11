import { type FormEvent, type KeyboardEvent, useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  BadgeCheck,
  ChevronLeft,
  ChevronRight,
  Copy,
  FileSearch,
  Plus,
  Send,
  Sparkles,
  X,
} from "lucide-react";
import { requestAnswer, type AnswerResponse, type Citation, type EvidenceRecord } from "./api";
import { prompts } from "./data";

type Turn = {
  id: string;
  question: string;
  state: "loading" | "answered" | "error";
  response?: AnswerResponse;
  error?: string;
};

type OpenSource = {
  response: AnswerResponse;
  index: number;
};

function Brand() {
  return (
    <div className="brand" aria-label="BankScope">
      <span className="brand-mark">B</span>
      <span>bankscope</span>
    </div>
  );
}

function Composer({
  value,
  onChange,
  onSubmit,
  disabled,
  compact = false,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (question?: string) => void;
  disabled: boolean;
  compact?: boolean;
}) {
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (value.trim() && !disabled) onSubmit();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (value.trim() && !disabled) onSubmit();
    }
  };

  return (
    <form className={`composer ${compact ? "composer-compact" : ""}`} onSubmit={submit}>
      <textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask about a bank's technology or operational risk…"
        rows={compact ? 1 : 3}
        aria-label="Research question"
        disabled={disabled}
      />
      <div className="composer-footer">
        <span>The bank and relevant filing are detected automatically.</span>
        <div className="send-group">
          {!compact && <small>Enter to send · Shift + Enter for a new line</small>}
          <button type="submit" className="send-button" disabled={!value.trim() || disabled} aria-label="Send question">
            <Send size={17} />
          </button>
        </div>
      </div>
    </form>
  );
}

function AnswerText({ response, onSource }: { response: AnswerResponse; onSource: (index: number) => void }) {
  const parts = response.answer.split(/(\[E\d+\])/g);

  return (
    <p className="answer-text">
      {parts.map((part, index) => {
        const label = /^\[(E\d+)\]$/.exec(part)?.[1];
        const citationIndex = label
          ? response.citations.findIndex((citation) => citation.label === label)
          : -1;
        return citationIndex >= 0 ? (
          <button key={`${part}-${index}`} className="citation" onClick={() => onSource(citationIndex)}>
            {label}
          </button>
        ) : (
          <span key={`${part}-${index}`}>{part}</span>
        );
      })}
    </p>
  );
}

function AssistantTurn({
  turn,
  copied,
  onCopy,
  onSource,
}: {
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
          <small>{turn.state === "loading" ? "Finding the bank and filing evidence…" : "Grounded in indexed filings"}</small>
        </div>
      </div>

      {turn.state === "loading" && (
        <div className="thinking-card" role="status">
          <span className="spinner" />
          <div><span /><span /><span /></div>
        </div>
      )}

      {turn.state === "error" && (
        <div className="error-card" role="alert">
          <strong>Answer could not be generated.</strong>
          <p>{turn.error}</p>
        </div>
      )}

      {turn.state === "answered" && turn.response && (
        <div className="answer-body">
          <AnswerText response={turn.response} onSource={(index) => onSource(turn.response!, index)} />
          <div className="answer-meta">
            {turn.response.ticker && <span className="bank-chip">{turn.response.ticker} detected</span>}
            <span>{turn.response.citations.length} {turn.response.citations.length === 1 ? "source" : "sources"}</span>
          </div>
          <div className="answer-actions">
            <button onClick={onCopy}><Copy size={14} /> {copied ? "Copied" : "Copy answer"}</button>
            {turn.response.citations.length > 0 && (
              <button onClick={() => onSource(turn.response!, 0)}><FileSearch size={14} /> View sources</button>
            )}
          </div>
        </div>
      )}
    </article>
  );
}

function metadataValue(evidence: EvidenceRecord | undefined, key: string): string {
  const value = evidence?.metadata?.[key];
  return typeof value === "string" || typeof value === "number" ? String(value) : "";
}

function citedEvidence(response: AnswerResponse, citation: Citation): EvidenceRecord | undefined {
  return response.evidence.find((item) => item.target_chunk_id === citation.target_chunk_id);
}

function SourcePanel({ source, onChange, onClose }: {
  source: OpenSource;
  onChange: (index: number) => void;
  onClose: () => void;
}) {
  const citation = source.response.citations[source.index];
  const evidence = citedEvidence(source.response, citation);
  const page = citation.display_page_start ?? citation.page_start;
  const section = citation.section_title || metadataValue(evidence, "section_title") || "Filing evidence";
  const filingDate = citation.filing_date || metadataValue(evidence, "filing_date");
  const document = evidence?.document || evidence?.evidence || evidence?.retrieval_text || "The source excerpt is not available.";

  return (
    <div className="source-overlay" role="dialog" aria-modal="true" aria-label="Answer sources">
      <button className="source-backdrop" onClick={onClose} aria-label="Close sources" />
      <aside className="source-panel">
        <div className="source-header">
          <div><span>Evidence</span><strong>{source.response.citations.length} {source.response.citations.length === 1 ? "source" : "sources"}</strong></div>
          <button className="icon-button" onClick={onClose} aria-label="Close sources"><X size={18} /></button>
        </div>
        <div className="source-tabs">
          {source.response.citations.map((item, index) => (
            <button key={item.label} className={index === source.index ? "active" : ""} onClick={() => onChange(index)}>{item.label}</button>
          ))}
        </div>
        <div className="source-card">
          <div className="source-title">
            <span>{citation.ticker || source.response.ticker || "SEC"}</span>
            <div><strong>{section}</strong><small>{[citation.record_type, page ? `page ${page}` : "", filingDate].filter(Boolean).join(" · ")}</small></div>
          </div>
          <p>{document}</p>
          {citation.source_url && <a href={citation.source_url} target="_blank" rel="noreferrer">Open original filing <ArrowRight size={14} /></a>}
        </div>
        <div className="source-navigation">
          <button disabled={source.index === 0} onClick={() => onChange(source.index - 1)}><ChevronLeft size={15} /> Previous</button>
          <span>{source.index + 1} of {source.response.citations.length}</span>
          <button disabled={source.index === source.response.citations.length - 1} onClick={() => onChange(source.index + 1)}>Next <ChevronRight size={15} /></button>
        </div>
      </aside>
    </div>
  );
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [openSource, setOpenSource] = useState<OpenSource | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const activeRequestRef = useRef<AbortController | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  const ask = async (suggestedQuestion?: string) => {
    const nextQuestion = (suggestedQuestion ?? question).trim();
    if (!nextQuestion || loading) return;

    const id = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}`;
    const sessionTicker = [...turns].reverse().find((turn) => turn.response?.ticker)?.response?.ticker ?? null;
    const controller = new AbortController();
    activeRequestRef.current = controller;
    setQuestion("");
    setLoading(true);
    setTurns((current) => [...current, { id, question: nextQuestion, state: "loading" }]);

    try {
      const response = await requestAnswer(nextQuestion, sessionTicker, controller.signal);
      setTurns((current) => current.map((turn) => turn.id === id ? { ...turn, state: "answered", response } : turn));
    } catch (error) {
      if (controller.signal.aborted) return;
      const message = error instanceof Error ? error.message : "Unexpected error while contacting the answer service.";
      setTurns((current) => current.map((turn) => turn.id === id ? { ...turn, state: "error", error: message } : turn));
    } finally {
      if (activeRequestRef.current === controller) {
        activeRequestRef.current = null;
        setLoading(false);
      }
    }
  };

  const reset = () => {
    activeRequestRef.current?.abort();
    activeRequestRef.current = null;
    setLoading(false);
    setQuestion("");
    setTurns([]);
    setOpenSource(null);
    setCopiedId(null);
  };

  const copyAnswer = async (turn: Turn) => {
    if (!turn.response) return;
    await navigator.clipboard.writeText(turn.response.answer);
    setCopiedId(turn.id);
    window.setTimeout(() => setCopiedId(null), 1400);
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <Brand />
        <div className="header-actions">
          <span className="corpus-status"><i /> 10 banks indexed</span>
          <button className="new-chat" onClick={reset}><Plus size={16} /> New conversation</button>
        </div>
      </header>

      {turns.length === 0 ? (
        <main className="welcome">
          <section className="welcome-content">
            <div className="eyebrow"><Sparkles size={14} /> Banking risk intelligence</div>
            <h1>Ask a question.<br />BankScope finds the bank.</h1>
            <p>Ask about technology and operational risk in indexed SEC filings. The relevant bank, document and evidence are selected automatically.</p>
            <Composer value={question} onChange={setQuestion} onSubmit={ask} disabled={loading} />
            <div className="suggestions">
              <span>Example questions</span>
              <div>
                {prompts.map((prompt) => (
                  <button key={prompt.label} onClick={() => void ask(prompt.text)}>
                    <span><strong>{prompt.label}</strong><small>{prompt.text}</small></span>
                    <ArrowRight size={16} />
                  </button>
                ))}
              </div>
            </div>
            <div className="trust-line"><BadgeCheck size={15} /> Answers are grounded only in indexed filing evidence.</div>
          </section>
        </main>
      ) : (
        <main className="conversation">
          <div className="conversation-list">
            {turns.map((turn) => (
              <section className="turn" key={turn.id}>
                <div className="user-turn"><span>You</span><p>{turn.question}</p></div>
                <AssistantTurn
                  turn={turn}
                  copied={copiedId === turn.id}
                  onCopy={() => void copyAnswer(turn)}
                  onSource={(response, index) => setOpenSource({ response, index })}
                />
              </section>
            ))}
            <div ref={endRef} />
          </div>
          <div className="conversation-composer">
            <Composer compact value={question} onChange={setQuestion} onSubmit={ask} disabled={loading} />
          </div>
        </main>
      )}

      {openSource && <SourcePanel source={openSource} onChange={(index) => setOpenSource({ ...openSource, index })} onClose={() => setOpenSource(null)} />}
    </div>
  );
}
