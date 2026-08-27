import { useState, type FormEvent } from "react";
import { File, Upload, X, CheckCircle, AlertCircle } from "lucide-react";
import { FileUpload } from "@/components/kokonutui/file-upload";
import { Button } from "./button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "./dialog";

interface FileUploadDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onUpload: (file: File, threadId?: string) => Promise<void>;
  threadId?: string;
  loading: boolean;
}

interface UploadStatus {
  type: "idle" | "uploading" | "success" | "error";
  message?: string;
  fileName?: string;
}

export function FileUploadDialog({ open, onOpenChange, onUpload, threadId, loading }: FileUploadDialogProps) {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<UploadStatus>({ type: "idle" });
  const [error, setError] = useState<string | null>(null);

  const allowedTypes = [
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/json",
  ];

  const allowedExtensions = [".pdf", ".txt", ".md", ".csv", ".doc", ".docx", ".xls", ".xlsx", ".json"];

  const maxSize = 10 * 1024 * 1024; // 10MB

  const selectFile = (selectedFile?: File) => {
    if (!selectedFile) return;

    setError(null);
    setStatus({ type: "idle" });

    // Validate file type by MIME type or extension
    const isTypeAllowed = allowedTypes.includes(selectedFile.type);
    const extension = selectedFile.name.toLowerCase().slice(selectedFile.name.lastIndexOf('.'));
    const isExtensionAllowed = allowedExtensions.includes(extension);

    if (!isTypeAllowed && !isExtensionAllowed) {
      setError("Unsupported file type. Please upload PDF, TXT, MD, CSV, DOC, DOCX, XLS, XLSX, or JSON files.");
      return;
    }

    // Validate file size
    if (selectedFile.size > maxSize) {
      setError("File size exceeds 10MB limit.");
      return;
    }

    setFile(selectedFile);
  };

  const handleRemoveFile = () => {
    setFile(null);
    setError(null);
    setStatus({ type: "idle" });
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();

    if (!file) return;

    setStatus({ type: "uploading", message: "Uploading...", fileName: file.name });
    setError(null);

    try {
      await onUpload(file, threadId);
      setStatus({ type: "success", message: "File uploaded successfully!", fileName: file.name });
      setFile(null);
      // Close dialog after a short delay to show success message
      setTimeout(() => onOpenChange(false), 1500);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to upload file.";
      setStatus({ type: "error", message: errorMessage, fileName: file.name });
      setError(errorMessage);
    }
  };

  const handleClose = () => {
    onOpenChange(false);
    handleRemoveFile();
    setStatus({ type: "idle" });
    setError(null);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <Dialog open={open} onOpenChange={(nextOpen) => (nextOpen ? onOpenChange(true) : handleClose())}>
      <DialogContent className="file-upload-dialog">
        <DialogHeader>
          <DialogTitle>Upload Document</DialogTitle>
          <DialogDescription>
            Upload a file for the LLM to use in its responses. Supported formats: PDF, TXT, MD, CSV, DOC, DOCX, XLS, XLSX, JSON. Max size: 10MB.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit}>
          <div className="file-upload-area">
            {status.type === "idle" && !file && (
              <FileUpload
                accept=".pdf,.txt,.md,.csv,.doc,.docx,.xls,.xlsx,.json"
                disabled={loading}
                hint="PDF, TXT, MD, CSV, DOC, DOCX, XLS, XLSX or JSON · up to 10MB"
                isInvalid={Boolean(error)}
                onFileSelect={selectFile}
              />
            )}

            {file && (
              <div className="file-upload-preview">
                <div className="file-info">
                  <File size={24} className="file-icon" />
                  <div className="file-details">
                    <span className="file-name">{file.name}</span>
                    <span className="file-size">{formatFileSize(file.size)}</span>
                  </div>
                  <Button type="button" variant="ghost" size="icon" onClick={handleRemoveFile} aria-label={`Remove ${file.name}`}>
                    <X size={16} />
                  </Button>
                </div>
              </div>
            )}

            {status.type === "uploading" && (
              <div className="upload-status uploading" role="status" aria-live="polite">
                <span className="spinner" />
                <span>{status.message} ({status.fileName})</span>
              </div>
            )}

            {status.type === "success" && (
              <div className="upload-status success" role="status" aria-live="polite">
                <CheckCircle size={16} />
                <span>{status.message}</span>
              </div>
            )}

            {error && (
              <div className="file-upload-error" role="alert">
                <AlertCircle size={16} />
                <span>{error}</span>
              </div>
            )}
          </div>

          <DialogFooter>
            <Button type="button" variant="outline" onClick={handleClose} disabled={loading || status.type === "uploading"}>
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={!file || loading || status.type === "uploading"}
            >
              <Upload size={16} /> Upload File
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
