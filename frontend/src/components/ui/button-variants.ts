import { cva } from "class-variance-authority";

export const buttonVariants = cva("ui-button", {
  variants: {
    variant: {
      default: "ui-button-default",
      secondary: "ui-button-secondary",
      ghost: "ui-button-ghost",
      outline: "ui-button-outline",
      destructive: "ui-button-destructive",
    },
    size: {
      default: "ui-button-md",
      sm: "ui-button-sm",
      icon: "ui-button-icon",
    },
  },
  defaultVariants: { variant: "default", size: "default" },
});
