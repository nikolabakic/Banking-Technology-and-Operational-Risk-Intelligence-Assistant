import { lazy, Suspense, useState } from "react";
import { Plus } from "lucide-react";
import { AttractButton } from "@/components/kokonutui/attract-button";
import { Tooltip, TooltipContent, TooltipTrigger } from "./tooltip";

const FileUploadDialog = lazy(() => import("./file-upload-dialog").then((module) => ({ default: module.FileUploadDialog })));

interface FileUploadButtonProps {
  threadId?: string;
  onUpload: (file: File, threadId?: string) => Promise<void>;
  disabled?: boolean;
  size?: "default" | "sm" | "icon";
  variant?: "default" | "outline" | "secondary" | "ghost";
}

export function FileUploadButton({
  threadId,
  onUpload,
  disabled = false,
  size = "icon",
  variant = "outline"
}: FileUploadButtonProps) {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [uploading, setUploading] = useState(false);

  const handleUpload = async (file: File, currentThreadId?: string) => {
    setUploading(true);
    try {
      await onUpload(file, currentThreadId);
    } finally {
      setUploading(false);
    }
  };

  const canUpload = !disabled && !uploading;

  return (
    <>
      <Tooltip>
        <TooltipTrigger asChild>
          <AttractButton
            type="button"
            size={size}
            variant={variant}
            onClick={() => setDialogOpen(true)}
            disabled={!canUpload}
            aria-label="Upload file"
          >
            <Plus size={size === "icon" ? 16 : 14} />
          </AttractButton>
        </TooltipTrigger>
        <TooltipContent>Upload document</TooltipContent>
      </Tooltip>

      {dialogOpen && (
        <Suspense fallback={<span className="sr-only" role="status">Opening upload dialog…</span>}>
          <FileUploadDialog
            open
            onOpenChange={setDialogOpen}
            onUpload={handleUpload}
            threadId={threadId}
            loading={uploading}
          />
        </Suspense>
      )}
    </>
  );
}
