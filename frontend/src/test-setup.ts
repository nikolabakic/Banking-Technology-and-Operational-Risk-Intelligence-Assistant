import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";
import { createElement, type ReactNode } from "react";

// Motion-specific props that should not be passed to DOM elements
const motionPropsToFilter = new Set([
  "initial",
  "animate",
  "exit",
  "transition",
  "variants",
  "whileHover",
  "whileTap",
  "whileInView",
  "whileDrag",
  "whileFocus",
  "whilePress",
  "layout",
  "layoutId",
  "drag",
  "dragConstraints",
  "dragElastic",
  "dragMomentum",
  "dragTransition",
  "transformTemplate",
  "onAnimationStart",
  "onAnimationComplete",
  "onDragStart",
  "onDrag",
  "onDragEnd",
  "onHoverStart",
  "onHoverEnd",
  "onTap",
  "onTapStart",
  "onTapCancel",
  "onPan",
  "onPanStart",
  "onPanEnd",
  "custom",
  "style",
] as const);

// Filter out motion-specific props
function filterMotionProps(props: Record<string, unknown>): Record<string, unknown> {
  const filtered: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(props)) {
    if (!motionPropsToFilter.has(key as typeof motionPropsToFilter extends Set<infer T> ? T : never)) {
      filtered[key] = value;
    }
  }
  return filtered;
}

// Mock motion module globally to prevent animation inline styles
// which change the DOM structure and break tests
vi.mock("motion/react", () => ({
  motion: {
    div: (props: { children?: ReactNode; [key: string]: unknown }) =>
      createElement("div", filterMotionProps(props), props.children),
    span: (props: { children?: ReactNode; [key: string]: unknown }) =>
      createElement("span", filterMotionProps(props), props.children),
    section: (props: { children?: ReactNode; [key: string]: unknown }) =>
      createElement("section", filterMotionProps(props), props.children),
    article: (props: { children?: ReactNode; [key: string]: unknown }) =>
      createElement("article", filterMotionProps(props), props.children),
    button: (props: { children?: ReactNode; [key: string]: unknown }) =>
      createElement("button", filterMotionProps(props), props.children),
    header: (props: { children?: ReactNode; [key: string]: unknown }) =>
      createElement("header", filterMotionProps(props), props.children),
    h1: (props: { children?: ReactNode; [key: string]: unknown }) =>
      createElement("h1", filterMotionProps(props), props.children),
    p: (props: { children?: ReactNode; [key: string]: unknown }) =>
      createElement("p", filterMotionProps(props), props.children),
    aside: (props: { children?: ReactNode; [key: string]: unknown }) =>
      createElement("aside", filterMotionProps(props), props.children),
    main: (props: { children?: ReactNode; [key: string]: unknown }) =>
      createElement("main", filterMotionProps(props), props.children),
  },
  AnimatePresence: (props: { children?: ReactNode }) => props.children,
  useAnimation: () => ({}),
  useAnimate: () => [() => {}],
}));

afterEach(() => cleanup());
