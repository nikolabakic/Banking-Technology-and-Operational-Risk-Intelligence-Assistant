/** Adapted from Kokonut UI's MIT-licensed Attract (magnet) Button. */
import { useRef, type MouseEvent, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type ButtonProps = React.ComponentProps<typeof Button> & { children: ReactNode };

export function AttractButton({ className, children, disabled, onMouseMove, onMouseLeave, ...props }: ButtonProps) {
  const ref = useRef<HTMLButtonElement>(null);
  const shellRef = useRef<HTMLDivElement>(null);

  const handleMove = (event: MouseEvent<HTMLButtonElement>) => {
    const bounds = ref.current?.getBoundingClientRect();
    if (bounds && !disabled) {
      const x = (event.clientX - bounds.left - bounds.width / 2) * 0.16;
      const y = (event.clientY - bounds.top - bounds.height / 2) * 0.16;
      if (shellRef.current) shellRef.current.style.transform = `translate3d(${x}px, ${y}px, 0)`;
    }
    onMouseMove?.(event);
  };
  const handleLeave = (event: MouseEvent<HTMLButtonElement>) => {
    if (shellRef.current) shellRef.current.style.transform = "translate3d(0, 0, 0)";
    onMouseLeave?.(event);
  };

  return (
    <div ref={shellRef} className="kokonut-magnet-shell">
      <Button ref={ref} className={cn("kokonut-magnet-button", className)} disabled={disabled} onMouseMove={handleMove} onMouseLeave={handleLeave} {...props}>{children}</Button>
    </div>
  );
}

export default AttractButton;
