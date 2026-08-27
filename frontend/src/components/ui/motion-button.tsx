import { forwardRef } from "react";
import { motion, type HTMLMotionProps, type MotionProps } from "motion/react";
import { cn } from "@/lib/utils";
import { buttonVariants } from "./button-variants";
import type { VariantProps } from "class-variance-authority";

export interface MotionButtonProps
  extends Omit<HTMLMotionProps<"button">, "children">,
    VariantProps<typeof buttonVariants> {
  children?: React.ReactNode;
  /** Custom whileHover animation */
  whileHover?: MotionProps["whileHover"];
  /** Custom whileTap animation */
  whileTap?: MotionProps["whileTap"];
  /** Custom transition */
  transition?: MotionProps["transition"];
}

const MotionButton = forwardRef<HTMLButtonElement, MotionButtonProps>(
  (
    {
      className,
      whileHover = { scale: 1.02, y: -1 },
      whileTap = { scale: 0.98 },
      transition = { type: "spring" as const, stiffness: 400, damping: 17 },
      children,
      variant,
      size,
      ...props
    },
    ref
  ) => {
    return <motion.button
      className={cn(buttonVariants({ variant, size }), className)}
      ref={ref}
      whileHover={whileHover}
      whileTap={whileTap}
      transition={transition}
      {...props}
    >
      {children}
    </motion.button>;
  }
);

MotionButton.displayName = "MotionButton";

export { MotionButton };
