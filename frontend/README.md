# BankScope frontend

Focused question-and-answer interface for the BankScope banking technology and
operational-risk assistant.

## Stack

- React 19 + TypeScript + Vite
- Geist variable font
- Lucide icons
- CSS design tokens using BankScope blue `#3459b1` and red `#ee413b`

## Run locally

Start the long-lived Python answer service in one terminal:

```powershell
cd frontend
npm run api
```

Then start Vite in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Vite proxies `/api` requests to `http://127.0.0.1:8000`. The API keeps the answer
pipeline loaded between questions, automatically resolves the bank from each question,
and uses the last resolved bank only as conversational context for follow-up questions.
It defaults to the schema-validated `AZURE_GPT_51_2025_1113` generation model. Override
it when needed with `npm run api -- --model MODEL_NAME`.

## Interface contract

- question submit → `POST /api/answer`
- answer status → `supported | ambiguous | unsupported`
- source chips and evidence drawer → `citations` plus hydrated `evidence`
- bank selection is deliberately absent; `SingleBankAnswerPipeline` resolves it automatically

## Custom logo

`Brand` in `src/App.tsx` currently uses a temporary `B` tile. When the final logo is
ready, add it under `public/` and replace the `.brand-mark` element with an image.
