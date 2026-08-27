import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import type * as React from "react";
import { motion, AnimatePresence } from "motion/react";
import { cn } from "@/lib/utils";
import { overlayFade, sheetAnimation, sheetLeftAnimation } from "./motion-presets";

const Sheet = DialogPrimitive.Root;
const SheetTrigger = DialogPrimitive.Trigger;
const SheetClose = DialogPrimitive.Close;

function SheetContent({
  className,
  children,
  side = "right",
  noAnimation = false,
  open,
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Content> & { side?: "left" | "right"; noAnimation?: boolean; open?: boolean }) {
  const animation = side === "left" ? sheetLeftAnimation : sheetAnimation;

  if (noAnimation) {
    return (
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="ui-sheet-overlay" />
        <DialogPrimitive.Content className={cn("ui-sheet-content", `ui-sheet-${side}`, className)} {...props}>
          {children}
          <DialogPrimitive.Close className="ui-sheet-close" aria-label="Close panel"><X size={18} /></DialogPrimitive.Close>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    );
  }

  return (
    <DialogPrimitive.Portal forceMount>
      <AnimatePresence>
        {open && (
          <>
            <DialogPrimitive.Overlay forceMount asChild>
              <motion.div key="overlay" className="ui-sheet-overlay" {...overlayFade} />
            </DialogPrimitive.Overlay>
            <DialogPrimitive.Content forceMount asChild {...props}>
              <motion.div
                key="content"
                className={cn("ui-sheet-content", `ui-sheet-${side}`, className)}
                initial={animation.initial}
                animate={animation.animate}
                exit={animation.exit}
                transition={animation.transition}
              >
                {children}
                <DialogPrimitive.Close className="ui-sheet-close" aria-label="Close panel"><X size={18} /></DialogPrimitive.Close>
              </motion.div>
            </DialogPrimitive.Content>
          </>
        )}
      </AnimatePresence>
    </DialogPrimitive.Portal>
  );
}

const SheetTitle = DialogPrimitive.Title;
const SheetDescription = DialogPrimitive.Description;

export { Sheet, SheetClose, SheetContent, SheetDescription, SheetTitle, SheetTrigger };
