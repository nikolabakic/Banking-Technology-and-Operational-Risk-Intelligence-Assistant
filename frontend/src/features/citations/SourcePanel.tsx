import { useEffect, useState, type ComponentType } from "react";
import { ArrowRight, ChevronDown, ChevronLeft, ChevronRight } from "lucide-react";
import { ApiError, loadCitationContext, type AnswerResponse, type CitationContext, type DocumentCitation, type FilingCitation } from "@/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/motion-sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export type OpenSource = {
  response: AnswerResponse;
  index: number;
  citation: FilingCitation | DocumentCitation;
};

function metadataValue(chunk: CitationContext["chunks"][number] | undefined, key: string): string {
  const value = chunk?.metadata?.[key];
  return typeof value === "string" || typeof value === "number" ? String(value) : "";
}

export default function SourcePanel({ source, onChange, onClose, markdown: Markdown }: {
  source: OpenSource;
  onChange: (index: number) => void;
  onClose: () => void;
  markdown: ComponentType<{ text: string }>;
}) {
  const citation = source.citation;
  const evidenceSources = source.response.citations.flatMap((item, index) => item.kind !== "web" ? [{ citation: item, index }] : []);
  const sourcePosition = evidenceSources.findIndex((item) => item.index === source.index);
  const [context, setContext] = useState<CitationContext | null>(null);
  const [error, setError] = useState<{ message: string; stale: boolean } | null>(null);
  const [showContextChunks, setShowContextChunks] = useState(false);

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
      <SheetContent open className="source-panel" aria-describedby="source-description">
        <div className="source-header"><div><span>Evidence viewer</span><SheetTitle>{evidenceSources.length} {evidenceSources.length === 1 ? "source" : "sources"}</SheetTitle><SheetDescription id="source-description">Canonical evidence context for this answer.</SheetDescription></div></div>
        <Tabs className="source-tabs-root" value={citation.citation_id} onValueChange={(value) => {
          const next = evidenceSources.find((item) => item.citation.citation_id === value);
          if (next) onChange(next.index);
        }}>
          <TabsList className="source-tabs" aria-label="Answer sources">
            {evidenceSources.map((item) => <TabsTrigger key={item.citation.citation_id} value={item.citation.citation_id}>{item.citation.label}</TabsTrigger>)}
          </TabsList>
        <TabsContent value={citation.citation_id} asChild>
        <ScrollArea className="source-scroll">
          <div className="source-card">
            <div className="source-title"><Badge>{citation.ticker || source.response.ticker || "SEC"}</Badge><div><strong>{section}</strong><small>{[citation.record_type, page ? `page ${page}` : "", filingDate].filter(Boolean).join(" · ")}</small></div></div>
            {!context && !error && <div className="source-loading" role="status"><span className="spinner" /> Loading canonical evidence…</div>}
            {error && <div className={`source-error ${error.stale ? "stale" : ""}`} role="alert"><strong>{error.stale ? "Source version changed" : "Source unavailable"}</strong><p>{error.message}</p></div>}
            {context?.chunks.some((chunk) => chunk.role !== "anchor") && <Button variant="ghost" size="sm" onClick={() => setShowContextChunks(!showContextChunks)} className="show-context-button" aria-expanded={showContextChunks}><ChevronDown size={14} className={showContextChunks ? "rotated" : ""} />{showContextChunks ? "Hide context" : "Show previous & next chunks"}</Button>}
            {context?.chunks.filter((chunk) => showContextChunks || chunk.role === "anchor").map((chunk) => <section className={`context-chunk ${chunk.role}`} key={`${chunk.target_chunk_id}-${chunk.role}`}><small>{chunk.role === "anchor" ? "Cited evidence" : `${chunk.role} context`}</small><Markdown text={chunk.document} /></section>)}
            {(context?.source_url || citation.source_url) && <Button asChild variant="outline" size="sm"><a href={context?.source_url || citation.source_url} target="_blank" rel="noreferrer">Open original filing <ArrowRight size={14} /></a></Button>}
          </div>
        </ScrollArea>
        </TabsContent>
        </Tabs>
        <div className="source-navigation">
          <Button variant="ghost" size="sm" disabled={sourcePosition <= 0} onClick={() => onChange(evidenceSources[sourcePosition - 1].index)}><ChevronLeft size={15} /> Previous</Button>
          <span>{sourcePosition + 1} of {evidenceSources.length}</span>
          <Button variant="ghost" size="sm" disabled={sourcePosition < 0 || sourcePosition === evidenceSources.length - 1} onClick={() => onChange(evidenceSources[sourcePosition + 1].index)}>Next <ChevronRight size={15} /></Button>
        </div>
      </SheetContent>
    </Sheet>
  );
}
