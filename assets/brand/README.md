# Canonical BankScope brand assets

**Status:** active design source and deterministic SVG exports.

```text
assets/brand/
├── bankscope.ai             # PDF-compatible Illustrator source
├── bankscope-wordmark.svg   # complete wordmark export
├── bankscope-mark.svg       # compact mark export
├── bankscope-target.svg     # assistant target export
└── README.md
```

```mermaid
flowchart LR
    AI[bankscope.ai] --> Export[scripts/export_logo_from_ai.mjs]
    Export --> Canonical[assets/brand/*.svg]
    Export --> Public[frontend/public/brand/*.svg]
    Public --> Vite[/brand/*.svg]
```

The Node exporter reads the embedded vector artwork, maps the BankScope palette, applies fixed
crops, and writes both canonical and web copies. It has no third-party dependency.

Run from the repository root:

```powershell
node scripts/export_logo_from_ai.mjs
```

The command must be deterministic: a second run should produce no Git diff. Keep public SVG URLs
stable when changing artwork because the frontend and its tests depend on them.

[Repository guide](../../README.md) · [Public copies](../../frontend/public/brand/README.md)

