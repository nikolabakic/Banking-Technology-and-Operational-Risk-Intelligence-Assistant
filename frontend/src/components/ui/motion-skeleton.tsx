import { motion, type HTMLMotionProps } from "motion/react";
import { cn } from "@/lib/utils";

function Skeleton({
  className,
  noAnimation = false,
  ...props
}: HTMLMotionProps<"div"> & { noAnimation?: boolean }) {
  if (noAnimation) {
    return <motion.div className={cn("ui-skeleton", className)} {...props} />;
  }

  return (
    <motion.div
      {...props}
      className={cn("ui-skeleton", className)}
      animate={{
        background: [
          "linear-gradient(90deg, #e6eef8 25%, #f2f7fc 50%, #e6eef8 75%)",
          "linear-gradient(90deg, #f2f7fc 25%, #e6eef8 50%, #f2f7fc 75%)",
        ],
      }}
      transition={{
        duration: 1.35,
        repeat: Infinity,
      }}
      style={props.style}
    />
  );
}

export { Skeleton };
