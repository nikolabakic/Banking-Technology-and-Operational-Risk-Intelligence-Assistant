import { useState, useRef, type ChangeEvent, type FormEvent, type MouseEvent as ReactMouseEvent, type DragEvent } from "react";
import { File, Upload, X, CheckCircle, AlertCircle } from "lucide-react";
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
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropZoneRef = useRef<HTMLDivElement>(null);
  const formRef = useRef<HTMLFormElement>(null);

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

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    selectFile(event.target.files?.[0]);
  };

  const handleRemoveFile = () => {
    setFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    setError(null);
    setStatus({ type: "idle" });
  };

  const handleDragEnter = (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(true);
  };

  const handleDrop = (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);

    selectFile(event.dataTransfer.files?.[0]);
  };

  const handleClick = (event: ReactMouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
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
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
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

        <form ref={formRef} onSubmit={handleSubmit}>
          <div className="file-upload-area">
            {status.type === "idle" && !file && (
              <div
                className={`file-upload-placeholder ${isDragging ? "dragging" : ""}`}
                ref={dropZoneRef}
                onClick={handleClick}
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
              >
                <Upload size={48} className="file-upload-icon" />
                <p className="file-upload-title">
                  {isDragging ? "Drop file here" : "Drag & drop file here or click to browse"}
                </p>
                <p className="file-upload-hint">Supported: PDF, TXT, MD, CSV, DOC, DOCX, XLS, XLSX, JSON</p>
                <p className="file-upload-hint">Max size: 10MB</p>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept=".pdf,.txt,.md,.csv,.doc,.docx,.xls,.xlsx,.json"
                  className="file-upload-input"
                  id="file-upload"
                  style={{ display: 'none' }}
                />
              </div>
            )}

            {file && (
              <div className="file-upload-preview">
                <div className="file-info">
                  <File size={24} className="file-icon" />
                  <div className="file-details">
                    <span className="file-name">{file.name}</span>
                    <span className="file-size">{formatFileSize(file.size)}</span>
                  </div>
                  <Button type="button" variant="ghost" size="icon" onClick={handleRemoveFile}>
                    <X size={16} />
                  </Button>
                </div>
              </div>
            )}

            {status.type === "uploading" && (
              <div className="upload-status uploading">
                <span className="spinner" />
                <span>{status.message} ({status.fileName})</span>
              </div>
            )}

            {status.type === "success" && (
              <div className="upload-status success">
                <CheckCircle size={16} />
                <span>{status.message}</span>
              </div>
            )}

            {status.type === "error" && (
              <div className="upload-status error">
                <AlertCircle size={16} />
                <span>{status.message}</span>
              </div>
            )}

            {error && (
              <div className="file-upload-error">
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
