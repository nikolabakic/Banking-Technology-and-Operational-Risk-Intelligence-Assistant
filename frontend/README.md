# BankScope frontend

Focused question-and-answer interface for exploring the latest indexed 10-K
filings from 10 leading U.S. banks, with answers grounded in verifiable filing evidence.

## Stack

- React 19 + TypeScript + Vite
- Geist variable font
- Lucide icons
- CSS design tokens using BankScope blue `#3459b1` and red `#ee413b`

## Run locally

Start the long-lived Python answer service in one terminal:

```powershell
cd frontend
npm.cmd run api
```

Then start Vite in a second terminal:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Vite proxies `/api` requests to `http://127.0.0.1:8000`. FastAPI keeps the answer
pipeline loaded, while SQLite persists threads, messages, bank context and citations.
The browser receives live pipeline stages over SSE. Historical sources are resolved
from the active canonical corpus when opened instead of being duplicated in the chat
database. The service defaults to `AZURE_GPT_51_2025_1113`; override it with
`npm.cmd run api -- --model MODEL_NAME`.

## Interface contract

- readiness check -> `GET /api/health`
- thread CRUD -> `/api/threads` and `/api/threads/{thread_id}`
- persisted history -> `GET /api/threads/{thread_id}/messages`
- streamed question -> `POST /api/threads/{thread_id}/stream`
- source context -> `GET /api/citations/{citation_id}/context`
- compatibility question -> `POST /api/answer`
- answer status -> `supported | ambiguous | unsupported`
- source chips carry persisted citation IDs; the drawer hydrates canonical evidence on demand
- bank selection is deliberately absent; `SingleBankAnswerPipeline` resolves it automatically

## Checks

```powershell
npm.cmd run lint
npm.cmd test
npm.cmd run build
```

## Brand assets

The header wordmark, assistant target and favicon mark are served from `public/brand/`.
Their editable SVG sources live at the repository root and can be regenerated from
the PDF-compatible Illustrator source with `node ../scripts/export_logo_from_ai.mjs`.
