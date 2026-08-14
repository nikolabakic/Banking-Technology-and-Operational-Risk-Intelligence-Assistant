# Public brand assets

**Status:** generated frontend copies.

Vite serves these files unchanged at `/brand/`:

| File | Use |
|---|---|
| `bankscope-wordmark.svg` | Application header |
| `bankscope-mark.svg` | Favicon and compact identity |
| `bankscope-target.svg` | Assistant message marker |

Do not edit these copies by hand. Run `node scripts/export_logo_from_ai.mjs` from the repository
root; the exporter reads [`assets/brand/bankscope.ai`](../../../assets/brand/README.md), writes the
canonical SVGs there, then updates these public copies. Frontend tests assert the public URLs.

[Frontend guide](../../README.md)

