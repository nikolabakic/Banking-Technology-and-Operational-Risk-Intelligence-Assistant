# BankScope snimci za finalni izveštaj

Paket je napravljen 31. avgusta 2026. stvarnim upitima kroz lokalni BankScope React interfejs i
aktivni FastAPI servis. Pitanja su preuzeta iz ručno verifikovanog evaluacionog skupa, a snimci nisu
makete. Za svaki primer sačuvani su odgovor sa citatima i prošireni prikaz dijagnostike.

## Preporučeni osnovni izbor

### 1. Tačna vrednost iz regulatorne tabele

**Pitanje:** What was Capital One Financial Corporation's CET1 capital ratio under the Basel III
standardized approach on December 31, 2025?

**Rezultat:** 14.3%; 2 izvora; status `supported`; Evidence audit `Passed`; izvršne provere `passed`.

- `01-capital-one-cet1-answer.png` — čist prikaz pitanja, odgovora i citata.
- `01-capital-one-cet1-diagnostics.png` — isti primer sa revizijom dokaza i dijagnostikom.
- `01-capital-one-cet1-source-evidence.png` — otvoren pregled kanonskog SEC dokaza.

Predlog potpisa slike: *BankScope pronalazi tačnu CET1 vrednost, vezuje odgovor za kanonske izvore
i prikazuje uspešnu reviziju dokaza i izvršne kontrole.*

### 2. Narativna sinteza rizika

**Pitanje:** How does Capital One manage cybersecurity and technology risk under its enterprise
risk framework?

**Rezultat:** objašnjen tretman u okviru operativnog rizika, model tri linije odbrane, životni ciklus
upravljanja i nadzor; 3 izvora; status `supported`; Evidence audit `Passed`; izvršne provere `passed`.

- `02-capital-one-cyber-risk-answer.png` — čist prikaz pitanja i dužeg, citiranog odgovora.
- `02-capital-one-cyber-risk-diagnostics.png` — kompletan odgovor, revizija dokaza i dijagnostika.

Predlog potpisa slike: *Narativna sinteza načina na koji Capital One integriše sajber i tehnološki
rizik u okvir upravljanja rizicima, sa citatima po materijalnim tvrdnjama.*

### 3. Poređenje dve banke

**Pitanje:** What were the Standardized CET1 capital ratios for Bank of America Corporation and
Citigroup on December 31, 2025?

**Rezultat:** Bank of America 11.40%, Citigroup 13.18%; odvojeni bankarski rezultati; 3 izvora;
status `supported`; Evidence audit `Passed`; izvršne provere `passed`.

- `04-bac-citi-cet1-comparison-answer.png` — čist prikaz sinteze i bankarski odvojenih rezultata.
- `04-bac-citi-cet1-comparison-diagnostics.png` — odgovor sa revizijom i kompletnom dijagnostikom.
- `04-bac-citi-cet1-comparison-source-evidence.png` — otvoren kanonski dokaz za regulatornu tabelu.

Predlog potpisa slike: *Bankarski izolovana višebankarska pretraga i sinteza Standardized CET1
pokazatelja za Bank of America i Citigroup.*

## Dodatni primer

`03-citi-jpm-operational-risk-*` prikazuje poređenje definicija operativnog rizika za Citi i
JPMorgan Chase. Koristan je kao dodatni kvalitativni primer, ali je za glavni tok izveštaja primer
04 sa tačnim vrednostima kompaktniji i vizuelno jasniji.

## Sažetak dijagnostike

| Primer | Ruta | Dokazi (početni → konačni) | Modelski zahtevi | Revizija dokaza | Izvršne provere |
|---|---|---:|---:|---|---|
| 01 Capital One CET1 | `domain_rag` | 4 → 4 | 3 | Passed | Passed |
| 02 Capital One cyber/technology risk | `domain_rag` | 3 → 3 | 3 | Passed | Passed |
| 03 Citi/JPM operational risk | `domain_rag` | 6 → 7 | 5 | Passed | Passed |
| 04 BAC/Citi CET1 | `domain_rag` | 6 → 6 | 5 | Passed | Passed |

Snimci prikazuju prihvaćenu baznu putanju sa isključenim eksperimentalnim Agentic RAG režimom.
Kontrole obuhvataju završetak cevovoda, očuvanje upita, izolaciju banaka, šemu plana, ugovor citata
i budžete akcija i zahteva.

## Reprodukcija i trag

- `capture-results.json` sadrži pitanja, očekivane i stvarne odgovore, dijagnostiku, vreme snimanja
  i URL lokalne niti.
- `capture-screenshots.mjs` ponavlja unos pitanja i snimanje kroz Playwright. Opcija
  `--reuse-existing` ponovo kadrira već sačuvane niti bez novog poziva modelu.
- Automatska očekivana provera prošla je za sva četiri primera.

Za finalni izveštaj je dovoljno umetnuti tri `*-diagnostics.png` snimka iz osnovnog izbora i jedan
`*-source-evidence.png` kao dokaz kanonske hidratacije. `*-answer.png` verzije su namenjene mestima
gde je važnija čitljivost odgovora od tehničke dijagnostike.
