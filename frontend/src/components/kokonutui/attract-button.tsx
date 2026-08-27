/** Adapted from Kokonut UI's MIT-licensed Attract (magnet) Button. */
import { motion } from "motion/react";
import { useRef, useState, type MouseEvent, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type ButtonProps = React.ComponentProps<typeof Button> & { children: ReactNode };

export function AttractButton({ className, children, disabled, onMouseMove, onMouseLeave, ...props }: ButtonProps) {
  const ref = useRef<HTMLButtonElement>(null);
  const [offset, setOffset] = useState({ x: 0, y: 0 });

  const handleMove = (event: MouseEvent<HTMLButtonElement>) => {
    const bounds = ref.current?.getBoundingClientRect();
    if (bounds && !disabled) {
      setOffset({
        x: (event.clientX - bounds.left - bounds.width / 2) * 0.16,
        y: (event.clientY - bounds.top - bounds.height / 2) * 0.16,
      });
    }
    onMouseMove?.(event);
  };
  const handleLeave = (event: MouseEvent<HTMLButtonElement>) => {
    setOffset({ x: 0, y: 0 });
    onMouseLeave?.(event);
  };

  return (
    <motion.div animate={offset} className="kokonut-magnet-shell" transition={{ type: "spring", stiffness: 320, damping: 22 }}>
      <Button ref={ref} className={cn("kokonut-magnet-button", className)} disabled={disabled} onMouseMove={handleMove} onMouseLeave={handleLeave} {...props}>{children}</Button>
    </motion.div>
  );
}

export default AttractButton;
