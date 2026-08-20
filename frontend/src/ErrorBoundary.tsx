import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = { children: ReactNode };
type State = { failed: boolean };

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { failed: false };

  static getDerivedStateFromError(): State {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("BankScope interface render failed", error, info.componentStack);
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <main className="recovery-screen" role="alert">
        <section className="recovery-card">
          <img src="/brand/bankscope-wordmark.svg" alt="BankScope" />
          <h1>The interface recovered from an invalid response.</h1>
          <p>Your saved conversations are still available. Reload the page to restore the workspace.</p>
          <button type="button" onClick={() => globalThis.location.reload()}>Reload BankScope</button>
        </section>
      </main>
    );
  }
}
