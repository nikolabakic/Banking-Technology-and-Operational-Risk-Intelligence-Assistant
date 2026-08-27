import type * as React from "react";
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

function SkeletonText({
  className,
  lines = 1,
  width = "100%",
  ...props
}: React.ComponentProps<"div"> & { lines?: number; width?: string | number }) {
  return (
    <div className={cn("flex flex-col gap-2", className)} {...props}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          className="h-4"
          style={{ width: i === lines - 1 ? width : "100%" }}
        />
      ))}
    </div>
  );
}

export { Skeleton, SkeletonText };
