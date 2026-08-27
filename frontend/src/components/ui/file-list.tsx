import { useEffect, useState } from "react";
import { File, X } from "lucide-react";
import { Button } from "./button";
import { LiquidGlassCard } from "@/components/kokonutui/liquid-glass-card";
import { Tooltip, TooltipContent, TooltipTrigger } from "./tooltip";
import { UserDocument, listDocuments, deleteDocument } from "@/api";

interface FileListProps {
  threadId: string;
  onFileDelete?: () => void;
}

const ALLOWED_TYPES: Record<string, string> = {
  "application/pdf": "PDF",
  "text/plain": "TXT",
  "text/markdown": "MD",
  "text/csv": "CSV",
  "application/msword": "DOC",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "DOCX",
  "application/vnd.ms-excel": "XLS",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "XLSX",
  "application/json": "JSON",
};

function getFileTypeIcon(contentType: string): string {
  const type = Object.entries(ALLOWED_TYPES).find(([key]) => key === contentType)?.[1];
  return type || "FILE";
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getFileIcon(contentType: string): React.ReactNode {
  const type = getFileTypeIcon(contentType);
  const icons: Record<string, React.ReactNode> = {
    PDF: <File size={16} className="file-icon pdf" />,
    TXT: <File size={16} className="file-icon txt" />,
    MD: <File size={16} className="file-icon md" />,
    CSV: <File size={16} className="file-icon csv" />,
    DOC: <File size={16} className="file-icon doc" />,
    DOCX: <File size={16} className="file-icon doc" />,
    XLS: <File size={16} className="file-icon xls" />,
    XLSX: <File size={16} className="file-icon xls" />,
    JSON: <File size={16} className="file-icon json" />,
    FILE: <File size={16} />,
  };
  return icons[type] || icons.FILE;
}

function getFileColor(contentType: string): string {
  const type = getFileTypeIcon(contentType);
  const colors: Record<string, string> = {
    PDF: "pdf",
    TXT: "txt",
    MD: "md",
    CSV: "csv",
    DOC: "doc",
    DOCX: "doc",
    XLS: "xls",
    XLSX: "xls",
    JSON: "json",
    FILE: "default",
  };
  return colors[type] || "default";
}

export function FileList({ threadId, onFileDelete }: FileListProps) {
  const [files, setFiles] = useState<UserDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    listDocuments(threadId, controller.signal)
      .then((documents) => {
        setFiles(documents);
        setError(null);
      })
      .catch((requestError: unknown) => {
        if (!controller.signal.aborted) {
          setError(requestError instanceof Error ? requestError.message : "Failed to load files");
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      });
    return () => controller.abort();
  }, [threadId]);

  const handleDelete = async (documentId: string) => {
    try {
      await deleteDocument(documentId);
      setFiles((prev) => prev.filter((f) => f.id !== documentId));
      onFileDelete?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete file");
    }
  };

  if (!threadId) return null;

  return (
    <div className="file-list">
      {loading && (
        <div className="file-list-loading">
          <span className="spinner" /> Loading files...
        </div>
      )}

      {error && (
        <div className="file-list-error">
          <span>{error}</span>
        </div>
      )}

      {files.length > 0 && (
        <div className="file-list-items">
          <h4 className="file-list-title">
            <File size={14} /> {files.length} {files.length === 1 ? "file" : "files"} attached
          </h4>
          <div className="file-items-grid">
            {files.map((file) => (
              <LiquidGlassCard key={file.id} className={`file-item file-item-${getFileColor(file.content_type)}`}>
                <div className="file-item-info">
                  {getFileIcon(file.content_type)}
                  <div className="file-item-details">
                    <span className="file-item-name">{file.filename}</span>
                    <span className="file-item-size">{formatFileSize(file.file_size)}</span>
                  </div>
                </div>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="file-item-delete"
                      onClick={() => handleDelete(file.id)}
                      aria-label={`Delete ${file.filename}`}
                    >
                      <X size={14} />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Delete file</TooltipContent>
                </Tooltip>
              </LiquidGlassCard>
            ))}
          </div>
        </div>
      )}

      {files.length === 0 && !loading && !error && (
        <div className="file-list-empty">
          <File size={16} />
          <span>No files attached to this conversation</span>
        </div>
      )}
    </div>
  );
}
