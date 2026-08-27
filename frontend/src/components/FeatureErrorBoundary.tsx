import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = { children: ReactNode; feature: string };
type State = { failed: boolean };

export class FeatureErrorBoundary extends Component<Props, State> {
  state: State = { failed: false };

  static getDerivedStateFromError(): State { return { failed: true }; }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`${this.props.feature} render failed`, error, info.componentStack);
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return <div className="feature-error" role="alert">{this.props.feature} is temporarily unavailable. The rest of the answer is still available.</div>;
  }
}
