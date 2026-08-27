/** Adapted from Kokonut UI's MIT-licensed Liquid Glass Card. */
import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export function LiquidGlassCard({ className, children, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("kokonut-glass-card", className)} {...props}><span className="kokonut-glass-shine" aria-hidden="true" /><div className="kokonut-glass-content">{children}</div></div>;
}

export default LiquidGlassCard;
