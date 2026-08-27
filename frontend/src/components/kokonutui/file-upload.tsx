/** Adapted from Kokonut UI's MIT-licensed File Upload component. */
import { UploadCloud } from "lucide-react";
import { motion } from "motion/react";
import { useRef, useState, type DragEvent } from "react";
import { cn } from "@/lib/utils";

interface FileUploadProps {
  accept?: string;
  disabled?: boolean;
  hint: string;
  isInvalid?: boolean;
  onFileSelect: (file: File) => void;
  className?: string;
}

export function FileUpload({ accept, disabled, hint, isInvalid, onFileSelect, className }: FileUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const select = (file?: File) => { if (file && !disabled) onFileSelect(file); };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    select(event.dataTransfer.files?.[0]);
  };

  return (
    <motion.div
      animate={{ scale: dragging ? 1.015 : 1 }}
      className={cn("kokonut-file-upload", dragging && "is-dragging", isInvalid && "is-invalid", className)}
      onClick={() => inputRef.current?.click()}
      onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
      onDragLeave={(event) => { event.preventDefault(); setDragging(false); }}
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
      role="button"
      tabIndex={disabled ? -1 : 0}
      onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") inputRef.current?.click(); }}
      aria-disabled={disabled}
    >
      <motion.span animate={{ y: dragging ? -4 : 0, rotate: dragging ? -4 : 0 }} className="kokonut-file-upload-icon" transition={{ type: "spring", stiffness: 360, damping: 20 }}>
        <UploadCloud size={34} />
      </motion.span>
      <strong>{dragging ? "Drop file here" : "Drag & drop file here"}</strong>
      <span>or click to browse</span>
      <small>{hint}</small>
      <input ref={inputRef} type="file" accept={accept} disabled={disabled} onChange={(event) => { select(event.target.files?.[0]); event.target.value = ""; }} className="sr-only" aria-label="Choose document to upload" />
    </motion.div>
  );
}

export default FileUpload;
