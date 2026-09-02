# BankScope — priprema za demonstraciju projekta

**Datum pripreme:** 2. septembar 2026.  
**Namena:** vodič za predstavljanje arhitekture, rada chatbota i kontrolisani live demo.  
**Preporučeno trajanje:** 20–30 minuta.

---

## 1. Najkraće objašnjenje projekta

BankScope je lokalni, single-user RAG asistent specijalizovan za istraživanje 10-K izveštaja deset
američkih banaka. On ne odgovara na pitanja o bankama samo iz „znanja modela”, već prvo pronalazi
relevantne delove lokalno indeksiranih SEC dokumenata, vraća kompletan kanonski dokaz, zatim od LLM-a
traži strukturiran odgovor i na kraju proverava citate, vlasništvo dokaza i numeričke činjenice.

Pored pitanja nad 10-K izveštajima, aplikacija podržava:

- razgovor o finansijama i tehnologiji bez nepotrebnog pokretanja RAG-a;
- kratkoročnu memoriju i prirodne follow-up upite;
- poređenje dve do četiri banke uz odvojenu pretragu za svaku banku;
- citiranu web pretragu za aktuelne ili promenljive podatke;
- deterministički kalkulator za aritmetiku;
- razgovor sa dokumentima koje korisnik uploaduje;
- lokalno čuvanje niti, poruka i metapodataka citata u SQLite-u.

### Elevator pitch od 30 sekundi

> BankScope je evidence-grounded istraživački asistent za bankarske 10-K izveštaje. Kombinuje
> semantičku pretragu iz Qdrant-a i leksičku BM25S pretragu, spaja rezultate pomoću RRF-a i svaki
> odgovor vezuje za kanonski SEC dokaz. Sistem ume da zapamti kontekst razgovora, poredi više banaka
> bez mešanja njihovih dokaza, koristi web samo za aktuelne informacije i odbija da izmisli odgovor
> kada dokaz nije dovoljan.

### Glavna poruka koju treba ponavljati

**LLM nije izvor činjenica; dokument je izvor činjenica. LLM služi za razumevanje pitanja i
sintezu odgovora, dok aplikacija kontroliše opseg, dokaze, citate, brojeve i budžete.**

---

## 2. Šta je provereno pre demoa

Stanje lokalnog okruženja provereno je 2. septembra 2026:

- FastAPI health endpoint vraća `{"status": "ready"}`;
- postoje lokalni `chunks.jsonl`, `tables.jsonl`, `embeddings.npz`, Qdrant skladište i njegov
  manifest;
- postoje Python virtualno okruženje i frontend `node_modules`;
- aktivni model je `AZURE_GPT_51_2025_1113`;
- `AGENTIC_RAG_ENABLED=false`, što je prihvaćena i bezbedna demo putanja;
- web search je uključen, provider je `auto`, a Tavily fallback je konfigurisan;
- Docling i `pypdfium2` su instalirani, pa je upload demo spreman i za PDF/Office formate;
- stvarni web upit **“Who is Citigroup's current CEO?”** prošao je danas sa rutom
  `web_research`, statusom `supported` i pet web citata;
- privremena nit korišćena za proveru obrisana je nakon testa.

Postojeći dokaz kvaliteta u repozitorijumu:

| Provera | Rezultat | Značenje za demo |
|---|---:|---|
| Application smoke, svih 10 banaka | 10/10 prošlo | Najmanje jedno podržano pitanje po banci radi sa citatima i ispravnim vlasništvom dokaza. |
| Single-bank generation | 30/30 bez greške; status accuracy 100% | Izabrana obična filing pitanja imaju jak evaluacioni oslonac. |
| Numeričke činjenice u generation evaluaciji | vrednost/jedinica/period/entity 100% | Precizna CET1 pitanja su najbolji izbor za live demo. |
| Semantic judge | correctness/completeness/groundedness 100% na 12 ocenjenih slučajeva | Narativni odgovori iz glavnog skupa prošli su semantičku proveru. |
| Conversation memory | 8/8 rewrite ugovora; Hit@5 porast 6/8 → 8/8 | Follow-up scenario iz ovog vodiča je eksplicitno pokriven evaluacijom. |
| Multi-bank comparison | 3/3 ukupno prošlo; 0 ownership povreda | Tri navedena poređenja su bezbedni demo kandidati. |
| Conversation routing | 53/55, odnosno 96,36%; gate prošao | Filing, web, kalkulator, direktan razgovor i out-of-scope rute su pokrivene. |
| UI QA snimci od 31. avgusta 2026. | sva 4 scenarija prošla | Glavni Capital One i BAC/Citi primeri su provereni kroz stvarni React/FastAPI interfejs. |

Napomena: nijedan generativni sistem nema matematičku garanciju da će svaki put dati identičnu
formulaciju. Zato su u nastavku prioritet dobila pitanja koja imaju ručno verifikovan gold odgovor,
uspešan evaluacioni rezultat ili stvarni UI QA trag. Za demo koristiti formulacije bez izmene.

---

## 3. Predloženi tok prezentacije

### Varijanta od 20 minuta

| Vreme | Šta pokazati | Glavna poruka |
|---:|---|---|
| 0–2 min | Problem i elevator pitch | Generički chatbot nije dovoljan za dokazive bankarske tvrdnje. |
| 2–5 min | Arhitektura i offline priprema korpusa | Dokumenti se unapred parsiraju, dele i indeksiraju uz očuvanje tabela. |
| 5–8 min | Jedno precizno CET1 pitanje | Hibridna pretraga, tačan broj i citat ka kanonskoj tabeli. |
| 8–11 min | Dva follow-up pitanja | Memorija razume referencu, ali prethodni odgovor ne tretira kao novi dokaz. |
| 11–14 min | Poređenje BAC i Citi | Svaka banka ima izolovanu pretragu, rezultat i citate. |
| 14–16 min | Web pitanje | Aktuelna činjenica ide na web, a ne u lokalni 10-K indeks. |
| 16–18 min | Upload test PDF-a | Dokument je thread-scoped izvor i dobija sopstvene citate. |
| 18–19 min | Nevezano pitanje | Sistem čuva finance/technology granicu i ne pokreće alat. |
| 19–20 min | Zaključak i ograničenja | Sistem radije apstinira nego da izmisli činjenicu. |

### Ako ima 30 minuta

Dodati:

- narativno pitanje o sajber/tehnološkom riziku;
- otvaranje **Sources** panela i kompletne tabele;
- otvaranje **Diagnostics** panela;
- kalkulator pitanje;
- pitanje koje nema podržan period kako bi se pokazao kontrolisani `unsupported` odgovor;
- kratko objašnjenje evaluacionih skupova i testova.

---

## 4. Kako sistem radi — objašnjenje za prezentaciju

## 4.1 Offline putanja: od SEC dokumenta do indeksa

```text
config/banks.yaml
      ↓
SEC download + data/filings.json
      ↓
sec2md parsiranje HTML 10-K dokumenta
      ↓
narativni chunks + opisi tabela + kompletne kanonske tabele
      ↓
Sentence Transformer embeddings
      ↓
Qdrant dense indeks + BM25S lexical indeks
```

1. `config/banks.yaml` definiše podržane banke, CIK brojeve, legalna imena i alias-e.
2. `scripts/download.py` pronalazi poslednji primarni 10-K i čuva originalni SEC HTML.
3. `scripts/build_corpus.py` koristi `sec2md==0.1.23` da iz HTML-a dobije strukturirane elemente.
4. Narativ se deli na ograničene, delimično preklopljene chunk-ove.
5. Tabele se **ne seku**. Za pretragu se pravi kompaktan opis, ali se kao dokaz čuva cela tabela.
6. `scripts/embed.py` pravi dense vektore za uređenu listu retrieval zapisa.
7. `scripts/build_qdrant.py` proverava hash-eve, dimenzije, broj tačaka i zatim gradi lokalni Qdrant.
8. BM25S indeks ostaje leksički sloj u aplikaciji; Qdrant je persistentni dense sloj.

Zašto su manifesti i hash-evi bitni: sprečavaju da aplikacija slučajno spoji novi korpus sa starim
embedding-om ili Qdrant indeksom. U slučaju neslaganja sistem prekida rad umesto da tiho koristi
zastarele podatke.

## 4.2 Online putanja: od pitanja do odgovora

```text
Korisničko pitanje + ograničena istorija
      ↓
conversation router
      ├─ direktan odgovor
      ├─ clarification
      ├─ out of scope
      ├─ filing research
      ├─ document research
      ├─ web research
      └─ calculator
```

Router prvo odlučuje **koji izvor je odgovarajući**:

- tvrdnja o podržanoj banci i njenom 10-K → lokalni filing RAG;
- aktuelna ili promenljiva činjenica → citirani web search;
- čista aritmetika → deterministički Decimal kalkulator;
- konceptualno finansijsko/tehnološko pitanje → direktan odgovor;
- pitanje o uploadovanom dokumentu → document research;
- nejasno pitanje → jedno kratko pojašnjenje;
- tema van finansija i tehnologije → `out_of_scope` bez pokretanja retrieval-a ili web-a.

Kod filing pitanja slede ovi koraci:

1. deterministički resolver prepoznaje banku iz legalnog imena, ticker-a ili alias-a;
2. follow-up se po potrebi pretvara u samostalni interni search upit;
3. originalno korisničko pitanje ostaje autoritativno;
4. dense i BM25S pretraga rade paralelno nad istim bankarskim opsegom;
5. rezultati se spajaju Reciprocal Rank Fusion algoritmom;
6. duplikati se uklanjaju po kanonskom target ID-u;
7. opis tabele se „hidrira” nazad u kompletnu tabelu;
8. LLM dobija samo ograničen skup dokaza i mora da vrati strogo strukturiran rezultat;
9. aplikacija proverava status, citate, bank ownership i numeričke tokene;
10. odgovor i metapodaci citata se atomski čuvaju u SQLite-u i šalju UI-ju preko SSE-a.

## 4.3 Zašto dense + BM25S + RRF

- Dense pretraga pronalazi semantički sličan sadržaj čak i kada pitanje i filing koriste različite
  izraze.
- BM25S je odličan za tačne termine, akronime, nazive metrika, godine i brojeve.
- RRF spaja rang-liste bez potrebe da dense i lexical skorovi budu na istoj skali.
- Ovaj „mixed” put je izmereni produkcioni baseline; full-Qdrant hybrid postoji samo za poređenje.

## 4.4 Tabele i kanonski dokaz

Ovo je jedna od najvažnijih tehničkih odluka projekta:

- `sec2md` tabela ostaje cela;
- retrieval pretražuje kratak opis tabele;
- hit nosi `table_id`/`target_chunk_id`;
- hydration vraća originalnu kompletnu Markdown tabelu;
- citat u odgovoru otvara upravo taj kanonski sadržaj.

Time se izbegava česta RAG greška u kojoj se red tabele odvoji od zaglavlja, jedinice ili perioda.

## 4.5 Memorija

SQLite čuva niti i poruke. Svaki novi threaded upit dobija ograničenu istoriju. Kada zbir sažetka i
transkripta pređe približno 12.000 tokena, stariji kompletni parovi se sažimaju, a najmanje šest
najnovijih parova ostaje doslovno.

Važna granica: istorija razgovora je **kontekst, a ne dokaz**. Ona pomaže da „it”, „that”, „what
about 2024?” ili „make it shorter” dobiju smisao, ali nova filing tvrdnja i dalje mora proći novu
pretragu i citiranje. Transformacija prethodnog odgovora može koristiti samo njegove postojeće
činjenice i citate; ne sme dodati novi broj, banku ili kvalifikator.

## 4.6 Poređenje više banaka

Za dve do četiri banke ne pravi se jedan pomešan evidence pool. Sistem:

1. deterministički prepoznaje uređenu listu banaka;
2. pravi bank-specific subquestion bez imena peer banke;
3. za svaku banku odvojeno radi Qdrant + BM25S + RRF;
4. za svaku banku odvojeno generiše i validira odgovor;
5. tek zatim sintetiše poređenje iz već validiranih bankarskih rezultata.

Ako jedna banka nema dovoljno dokaza, ostale ne propadaju. Odgovor postaje `partial`, a dokaz jedne
banke nikada ne sme biti pripisan drugoj.

## 4.7 Web search i kalkulator

Web search je namenjen podacima koji mogu da se promene: trenutni CEO, cena akcije, najnovija vest
ili regulatorna objava. Provider `auto` prvo pokušava OpenAI Responses `web_search`, a zatim koristi
Tavily ako gateway ne podržava Responses endpoint. Rezultat bez validnih HTTP(S) citata se ne
predstavlja kao podržan aktuelan odgovor.

Kalkulator parsira ograničen aritmetički izraz i računa pomoću `Decimal`. Ne koristi `eval` ili
`exec`, odbija funkcijske pozive, imena, atribute, prevelike eksponente, deljenje nulom i ne-finite
vrednosti.

## 4.8 Uploadovani dokumenti

Upload je vezan za konkretnu conversation thread i ograničen je na 10 MB po fajlu. Podržani su PDF,
TXT, Markdown, CSV, JSON, DOC/DOCX i XLS/XLSX. Tekstualni PDF može se parsirati lokalnim lightweight
parserom; Docling obrađuje kompleksnije Office, layout-aware i skenirane dokumente kada je instaliran.
Parsiran sadržaj je ograničen na 200.000 karaktera.

Dokument dobija sopstveno vlasništvo izvora (`source_kind=user_document`). Njegovi citati se otvaraju
iz thread-local SQLite sadržaja, a ne iz SEC korpusa. Sistem podržava i eksplicitno poređenje
uploadovanog dokumenta sa filing-om podržane banke, pri čemu ta dva skupa dokaza ostaju razdvojena.

## 4.9 Agentic RAG

Bounded Agentic RAG postoji kao eksperimentalni, aditivni sloj, ali je podrazumevano i trenutno
isključen: `AGENTIC_RAG_ENABLED=false`. Kada je uključen, baseline retrieval uvek ide prvi, a zatim
svaka banka dobija strogo ograničenu petlju sa `search_hybrid`, `search_exact`, `read_context` ili
`finish`. Runtime poseduje ticker, accession, target ID-eve, limite i budžete.

Za demo koristiti isključeni režim. To predstavlja prihvaćeni produkcioni baseline i poklapa se sa
UI QA snimcima od 31. avgusta.

---

## 5. Folderi i važni fajlovi

## 5.1 Root repozitorijuma

| Putanja | Uloga |
|---|---|
| `README.md` | Glavni tehnički pregled, quick start, arhitektura i poznata ograničenja. |
| `AGENTS.md` | Pravila rada u repozitorijumu; naročito važni provenance, test i Git ugovori. |
| `pyproject.toml` | Python 3.13 paket, dependency rasponi, extras (`dev`, `llm`, `docling`), Pytest i Ruff podešavanja. |
| `uv.lock` | Zaključane Python zavisnosti za reproduktivno okruženje. |
| `.env.example` | Bezbedan spisak konfiguracionih promenljivih bez tajni. |
| `.env` | Lokalni modeli, ključevi i podešavanja; nikada se ne commit-uje niti prikazuje na demou. |
| `start-app.ps1` | Windows launcher: bira `.venv` Python, proverava npm, čeka API health, pokreće Vite i gasi child procese na izlazu. |
| `config/` | Statički, verzionisan domen podržanih banaka. |
| `data/` | Filing manifest, lokalno generisani korpus/indeksi, evaluacioni ugovori i SQLite stanje. |
| `src/` | Reusable produkcioni Python kod. |
| `scripts/` | Tanki CLI entry point-i koji orkestriraju kod iz `src/bankscope`. |
| `frontend/` | React/TypeScript/Vite korisnički interfejs. |
| `tests/` | Aktivni Python unit, integration i contract testovi. |
| `docs/` | ADR odluke, roadmap, evaluacioni i QA izveštaji. |
| `notebooks/` | Colab GPU workflow za full-corpus embeddings i retrieval evaluaciju. |
| `assets/` | Kanonski brand i screenshot resursi. |
| `sandbox/` | Arhiviran/superseded kod; nije deo aktivne aplikacije. |
| `experiments/` | Eksperimentalni rad; ne tretira se kao produkcioni put. |
| `artifacts/` | Lokalno generisani bundle-i i izlazi; nisu aplikacioni izvor istine. |
| `.venv/`, `node_modules/`, cache i temp folderi | Lokalno razvojno okruženje i prolazni fajlovi; ne objašnjavati kao deo domenske arhitekture. |

## 5.2 `config/`

`config/banks.yaml` je jedini izvor istine za podržane banke. Trenutno su uključeni:

- JPMorgan Chase (`JPM`)
- Bank of America (`BAC`)
- Citigroup (`C`)
- Capital One (`COF`)
- State Street (`STT`)
- PNC (`PNC`)
- Truist (`TFC`)
- Goldman Sachs (`GS`)
- Ally Financial (`ALLY`)
- Live Oak Bancshares (`LOB`)

Svaki zapis ima zero-padded SEC CIK, legalno ime, alias-e i `enabled` zastavicu. Isti registry koristi
download, bank resolver i izgradnja indeksa, pa se lista banaka ne duplira u kodu.

## 5.3 `data/`

| Putanja | Uloga |
|---|---|
| `data/filings.json` | Verzionski manifest preuzetih 10-K izveštaja. |
| `data/raw/sec/...` | Originalni SEC HTML po CIK-u i accession broju; lokalno generisan. |
| `data/processed/chunks.jsonl` | Retrieval zapisi: narativni chunk-ovi i opisi tabela. |
| `data/processed/tables.jsonl` | Kompletne parser-emitted tabele kao kanonski dokaz. |
| `data/processed/lexical_glossary_locators_v1.jsonl` | Mali BM25-only locator zapisi za definicije, koji upućuju na stvarnu parent tabelu. |
| `data/processed/manifest.json` | Provenance korpusa: parser/config/source hash-evi. |
| `data/processed/embeddings.npz` | Uređeni `float32` dense vektori sa model i source metadata ugovorom. |
| `data/processed/qdrant/` | Persistentno lokalno Qdrant skladište. |
| `data/processed/qdrant_manifest.json` | Veza između kolekcije, korpusa, modela, dimenzije i broja tačaka. |
| `data/evaluation/*.jsonl` | Verzionska pitanja, qrels, memory/routing/challenge ugovori. |
| `data/evaluation/results/` | Rezultati konkretnih evaluacionih pokretanja. |
| `data/local/bankscope_chat.db` | Lokalni thread-ovi, poruke, upload-i i metapodaci citata. |

## 5.4 `src/bankscope/`

| Putanja | Uloga |
|---|---|
| `api.py` | FastAPI modeli i rute, SSE status/heartbeat/terminal eventi, upload endpoint-i i mapiranje grešaka. |
| `io.py` | Atomski JSONL upis, SHA-256 i stroga validacija embedding arhive. |
| `config/settings.py` | Pydantic Settings za `.env`, putanje, modele, SEC identitet, web i feature flag-ove. |
| `sec/company_registry.py` | Validacija banke, ticker-a, CIK-a, alias-a i duplikata. |
| `sec/bank_resolver.py` | Determinističko prepoznavanje jedne ili više banaka iz pitanja i session scope-a. |
| `parsing/corpus.py` | Redosled elemenata, uklanjanje navigacije, narrative chunking, stable ID i corpus validacija. |
| `parsing/tables.py` | Čitanje Markdown grid tabela, klasifikacija, naslov/jedinica, opis i kanonski table record. |
| `retrieval/hybrid_retriever.py` | NumPy dense baseline, BM25S, RRF, filteri i hydration. |
| `retrieval/qdrant_retriever.py` | Persistentni Qdrant upiti i validacija Qdrant manifesta. |
| `retrieval/mixed_retriever.py` | Prihvaćeni Qdrant-dense + BM25S + application-RRF put i bank-balanced interleaving. |
| `retrieval/glossary_locators.py` | Leksički locator-i koji definiciju pronalaze, ali citat vraćaju na kanonsku tabelu. |
| `generation/conversation.py` | LangGraph conversation router, stroge action šeme i deterministički fallback/policy validation. |
| `generation/memory.py` | Sažimanje starijih kompletnih parova i očuvanje referenci/preferencija. |
| `generation/contextualizer.py` | Pretvaranje follow-up pitanja u samostalan search upit uz očuvanje originala. |
| `generation/query_planner.py` | Validacija rewrite-a, bank-specific decomposition i više aspect upita za whole-filing summary. |
| `generation/pipeline.py` | Dugovečni centralni orkestrator za rute, retrieval, generaciju, poređenja, dokumente i diagnostics. |
| `generation/answer_generator.py` | Stroge output šeme, numeric facts, citati, support validacija i lokalno renderovanje odgovora. |
| `generation/comparison_generator.py` | Sinteza samo nad validiranim, bank-owned rezultatima. |
| `generation/agentic.py` | Eksperimentalne bounded akcije, state, verifier i kanonsko proširenje konteksta. |
| `chat/store.py` | SQLite schema/migracije, thread CRUD, poruke, bounded memory, upload i atomsko čuvanje turn-a. |
| `chat/sources.py` | Ponovno otvaranje kanonskog teksta/tabele i odbijanje zastarelih citata. |
| `llm/client.py` | Konstrukcija OpenAI-compatible klijenta bez širenja gateway detalja po aplikaciji. |
| `llm/model_access.py` | Interni corporate gateway adapter i fallback environment mapping. |
| `tools/web_search.py` | Provider-neutral OpenAI/Tavily pretraga, URL validacija, deduplikacija i cited result contract. |
| `tools/calculator.py` | Bezbedan bounded AST + Decimal kalkulator. |
| `evaluation/retrieval_metrics.py` | Hit/rank/MRR i multi-bank evidence-group metrike. |
| `evaluation/answer_metrics.py` | Status, entity, value, unit, period i citation metrike. |
| `evaluation/semantic_judge.py` | Offline gold judge i savetodavni runtime evidence audit. |

## 5.5 `scripts/` — šta radi svaka skripta

| Skripta | Uloga |
|---|---|
| `download.py` | Preuzima poslednji primarni 10-K za jednu ili sve konfigurisane banke i atomski ažurira manifest. |
| `build_corpus.py` | Pokreće aktivni `sec2md` parser, pravi narativne chunk-ove, opise/kanonske tabele i provenance manifest. |
| `build_glossary_locators.py` | Regeneriše samo BM25 glossary locator-e bez ponovne izgradnje celog korpusa. |
| `embed.py` | Pravi normalizovane query/document embedding-e i proverava da ništa nije tiho skraćeno. |
| `build_qdrant.py` | Validira korpus i embeddings pa gradi persistentnu lokalnu Qdrant kolekciju i manifest. |
| `build_colab_bundle.py` | Pravi minimalan hash-audited ZIP za podržani Colab GPU notebook. |
| `search.py` | CLI inspekcija dense, BM25 ili hybrid retrieval-a bez generisanja odgovora. |
| `answer.py` | Jedan CLI single-bank ili multi-bank odgovor kroz reusable pipeline. |
| `serve_api.py` | Učitava dugovečne servise i pokreće FastAPI/Uvicorn sa SQLite-em i opcionim web providerima. |
| `smoke_qdrant.py` | Brza provera da lokalna Qdrant kolekcija može da odgovori na mali upit. |
| `smoke_answers.py` | Fiksni application-pipeline smoke: po jedno grounded pitanje za svaku od deset banaka. |
| `evaluate.py` | Frozen retrieval evaluacija, backend poređenje, paritet i quality gate-ovi. |
| `evaluate_answers.py` | Single-bank generation evaluacija sa determinističkim metrikama i opcionim semantic judge-om. |
| `evaluate_conversation_memory.py` | Poredi stateless i contextualized retrieval za follow-up upite i proverava izolaciju niti. |
| `evaluate_conversation_routing.py` | Testira samo semantic routing bez retrieval-a i answer generation-a. |
| `evaluate_comparisons.py` | Testira tri frozen multi-bank pitanja, evidence groups, semantiku i citation ownership. |
| `evaluate_agentic_rag.py` | Poredi baseline i eksperimentalni bounded agentic retrieval na frozen 12-case izazovu. |
| `evaluate_evidence_audit_challenge.py` | Pokreće odvojeni 10-case izazov za savetodavni evidence audit. |
| `benchmark_query_embeddings.py` | Meri warm-up i ponovljenu latenciju query encoder-a. |
| `probe_generation_json.py` | Mali sintetički test JSON-mode kompatibilnosti gateway-a, bez filing podataka. |
| `export_logo_from_ai.mjs` | Deterministički generiše kanonske i javne SVG brand asset-e. |

Skripte su namerno tanke. Poslovna pravila ostaju u `src/bankscope`, kako CLI, API i testovi ne bi
imali tri različite implementacije iste logike.

## 5.6 `frontend/`

| Putanja | Uloga |
|---|---|
| `frontend/package.json` | React/Vite komande i zaključani dependency opsezi. |
| `src/main.tsx` | Browser entry point, router i top-level recovery boundary. |
| `src/ErrorBoundary.tsx` | Poslednja zaštita od praznog ekrana pri render grešci. |
| `src/App.tsx` | Thread state, slanje pitanja, streaming progress, upload, dialogs, comparison i Markdown odgovor. |
| `src/api.ts` | Tipizirani REST/SSE klijent i runtime validacija backend payload-a. |
| `src/data.ts` | Statički predlozi pitanja na početnom ekranu. |
| `src/features/citations/SourcePanel.tsx` | Lazy panel za kanonski evidence prikaz. |
| `src/components/ui/` | Mali ponašajno neutralni Radix-based UI primitive-i. |
| `src/styles/` i `src/index.css` | Design tokeni, teme, motion, reduced-motion i application stilovi. |
| `src/App.test.tsx`, `src/api.test.ts` | User-visible ponašanje i SSE/API ugovori. |
| `tests/accessibility/workspace.spec.ts` | Playwright + axe provere ključnih workspace tokova. |
| `public/brand/` | SVG wordmark, target i mark koje browser direktno servira. |

SSE omogućava da korisnik vidi faze poput embedding-a, retrieval-a, generation-a, validation-a i
synthesis-a dok dugotrajan odgovor još traje. Klijent ume da obradi fragmentisane LF/CRLF blokove,
heartbeat komentare i malformed međuevente bez rušenja celog interfejsa.

## 5.7 `tests/`, `docs/`, `notebooks/`, `sandbox/`

- `tests/` prati vlasništvo modula: settings, parser, retrieval, generation, API, SQLite, upload,
  web, calculator, comparisons, memory i frontend ugovore.
- `tests/fixtures/manual_upload_test.pdf` je deterministički, nepoverljiv demo dokument; očekivani
  odgovori su u susednom Markdown fajlu.
- `docs/decisions/` čuva ADR zapise koji objašnjavaju zašto su izabrani parser, Qdrant, mixed
  retrieval, memory i bank-isolated comparison.
- `docs/final-report-assets/qa-demo-2026-08-31/` sadrži stvarne UI snimke, pitanja, rezultate i
  Playwright trag, a ne makete.
- `notebooks/BankScope_GPU_Evaluation_Colab.ipynb` je podržan GPU put za full-corpus embedding build.
- `sandbox/` sadrži završene eksperimente i zastarele implementacije; nije aktivni proizvodni kod.

---

## 6. Glavni live demo — copy/paste scenario

Za maksimalnu pouzdanost pitanja ispod postavljati na engleskom, tačno kako su zapisana. Objašnjenje
publici može biti na srpskom. Pre početka otvoriti novu conversation thread.

## 6.1 Otvaranje i capability pitanje

**Pitanje 1**

```text
Which banks do you support?
```

Očekivano ponašanje:

- direktna `capability` ruta, bez filing retrieval-a;
- lista dolazi iz server-owned `config/banks.yaml`, ne iz improvizovane LLM liste;
- treba da navede deset podržanih banaka.

Šta reći publici: „Čak ni capability odgovor ne prepuštamo memoriji modela; registry je izvor
istine.”

## 6.2 Precizna numerička činjenica i kanonska tabela

**Pitanje 2 — preporučeni glavni primer**

```text
What was Capital One Financial Corporation's CET1 capital ratio under the Basel III standardized approach on December 31, 2025?
```

Očekivani sadržaj: **14.3%**.  
Očekivano ponašanje: `supported`, numeric answer, filing citati, Evidence audit `Passed` ili
savetodavno `unavailable` bez promene validnog odgovora.

Posle odgovora:

1. otvoriti jedan source chip;
2. pokazati da se otvara kompletna regulatorna tabela, ne samo isečen snippet;
3. otvoriti **Diagnostics**;
4. pokazati route, evidence count, očuvanje upita i execution checks;
5. naglasiti da `execution checks passed` znači da je cevovod poštovao ugovor, a ne nezavisnu
   revizorsku garanciju svake tvrdnje.

Ovo pitanje je prošlo stvarni UI QA 31. avgusta 2026. i ima sačuvane answer, diagnostics i source
snimke.

## 6.3 Memorija — pitanja jedno za drugim

Za ovaj scenario otvoriti **novu nit**, da bude jasno da nema skrivenog konteksta.

**Pitanje 3a**

```text
What was Bank of America Corporation's CET1 capital ratio under the Advanced approaches on December 31, 2025?
```

Očekivani sadržaj: **12.8%**.

**Pitanje 3b — bez ponavljanja banke, metrike i datuma**

```text
How does that compare with the Standardized approach?
```

Očekivani sadržaj:

- Bank of America;
- CET1;
- datum 31. decembar 2025;
- Standardized vrednost **11.4%**;
- Advanced 12.8% je viši za **1.4 procentna poena**, ako model izračuna razliku.

Ovaj tačan follow-up je deo memory evaluacije. Contextualized retrieval je u celom skupu podigao
Hit@5 sa 6/8 na 8/8, a svih osam rewrite ugovora je prošlo.

**Pitanje 3c — transformacija prethodnog grounded odgovora**

```text
Summarize the previous answer in one sentence and keep its citations.
```

Očekivano ponašanje: kratka `contextual_transform` poruka bez novih činjenica, banaka, brojeva ili
citata. Time se pokazuje razlika između „koristi prethodni odgovor za formatiranje” i „koristi ga kao
dokaz za novu tvrdnju”.

Ako želite kraću memory demonstraciju, završite posle 3b.

## 6.4 Poređenje dve banke

**Pitanje 4 — najstabilniji multi-bank primer**

```text
What were the Standardized CET1 capital ratios for Bank of America Corporation and Citigroup on December 31, 2025?
```

Očekivani sadržaj:

- Bank of America Corporation: **11.4%**;
- Citigroup: **13.18%**;
- Citigroup je viši za **1.78 procentnih poena**, ako model doda razliku;
- odvojeni bank results i bank-owned citati.

Ovaj upit je prošao frozen multi-bank evaluaciju i stvarni UI QA 31. avgusta 2026. sa statusom
`supported`, tri izvora, Evidence audit `Passed` i bez citation ownership povreda.

Šta reći: „Sistem ne pretražuje jedan zajednički bazen pa nagađa koji dokaz pripada kojoj banci.
Pretraga i validacija se rade posebno za BAC i Citi, a sinteza dolazi tek na kraju.”

## 6.5 Web search

**Pitanje 5 — provereno istog dana**

```text
Who is Citigroup's current CEO?
```

Očekivano ponašanje:

- route `web_research`;
- `dialog_act=web_answer`;
- status `supported`;
- odgovor sa klikabilnim web izvorima;
- ne koristi lokalni filing RAG samo zato što je pomenut Citi.

U preflight proveri 2. septembra 2026. odgovor je naveo Jane Fraser i vratio pet citata. Pošto je
„current CEO” vremenski promenljiva činjenica, na demou proveravati rutu i izvore, a ne učiti odgovor
napamet.

Rezervni web promptovi, čije je rutiranje prošlo evaluaciju:

```text
What is Citi's share price today?
```

```text
What is the latest news about JPMorgan?
```

Izbegavati `What is the latest stable Python version?` kao glavni demo upit: to je jedan od dva
promašaja u routing evaluaciji i jednom je završio kao clarification umesto web search-a.

## 6.6 Obično konceptualno pitanje bez RAG-a

**Pitanje 6**

```text
Explain retrieval-augmented generation.
```

Očekivano ponašanje: `direct_response`, bez filing izvora i bez web-a. Ovo pokazuje da sistem nije
„retrieval za svaku poruku”, već bira najjeftiniji odgovarajući put.

Alternative koje su takođe prošle routing evaluaciju:

```text
What is operational risk in banking?
```

```text
What is a zero-trust network architecture?
```

## 6.7 Deterministički kalkulator

**Pitanje 7**

```text
What is 17.5% of 2,400?
```

Očekivani rezultat: **420**.  
Očekivana ruta: `calculator`, bez filing retrieval-a i bez web-a.

Šta reći: „Za čistu aritmetiku ne tražimo od LLM-a da mentalno računa. Model bira alat, ali Decimal
kalkulator daje rezultat.”

## 6.8 Nevezano pitanje

**Pitanje 8**

```text
Daj mi recept za pitu sa jabukama.
```

Očekivano ponašanje: kratak `out_of_scope` odgovor koji usmerava korisnika na finansije i
tehnologiju; nema filing retrieval-a, web-a ni kalkulatora. Ovaj tačan srpski slučaj je deo routing
evaluacije i prošao je.

Šta reći: „Sistem je razgovoran, ali ima namerno ograničen profesionalni domen. Stari bankarski
kontekst ne može nevezano pitanje da učini relevantnim.”

## 6.9 Upload dokumenta

Otvoriti novu nit i uploadovati:

```text
tests/fixtures/manual_upload_test.pdf
```

Sačekati da UI pokaže uspešno parsiran attachment. Zatim postaviti sledeća pitanja.

**Pitanje 9a**

```text
What institution and reporting period does the document cover?
```

Očekivano: **North River Bank, fiscal year 2025**.

**Pitanje 9b**

```text
How many operational incidents occurred, and what were the total operational losses?
```

Očekivano: **14 incidents and USD 2.75 million**.

**Pitanje 9c**

```text
Did recovery performance meet its target? Show the relevant figures.
```

Očekivano: **da; prosečno vreme oporavka 37 minuta naspram cilja od 45 minuta**.

**Pitanje 9d — provera kontrolisane apstinencije**

```text
Which requested fact is not stated: customer count, operational losses, or service availability?
```

Očekivano: **customer count is not stated**; asistent ne sme da izmisli broj klijenata.

Na source chip-u pokazati da citat pripada uploadovanom dokumentu, a ne SEC korpusu.

Rezervna upload pitanja:

| Pitanje | Očekivano |
|---|---|
| `What was critical-services availability?` | 99.97%. |
| `How many severe cyber incidents and high-risk third-party providers were reported?` | 0 severe cyber incidents; 3 high-risk providers. |
| `How much did the bank invest in technology resilience?` | USD 18.4 million. |
| `What is the main remediation priority?` | Privileged-access monitoring. |
| `When is the next resilience exercise scheduled?` | October 2026. |
| `Summarize the document in three bullet points without adding external facts.` | Tri kratke, dokumentom podržane stavke. |

---

## 7. Rezervna filing pitanja sa jakim dokazom

Ako glavni prompt ne može da se pošalje zbog privremenog API problema, ne menjati nasumično temu.
Otvoriti novu nit i koristiti jedno od sledećih evaluiranih pitanja.

| Pitanje | Očekivani sadržaj | Dokaz pouzdanosti |
|---|---|---|
| `How does Ally Financial define operational risk in its 2025 Form 10-K?` | Rizik gubitka/štete zbog neadekvatnih ili neuspešnih procesa/sistema, ljudskih faktora ili spoljnih događaja; inherentan risk-generating aktivnostima. | Application smoke prošao; dva direktna citata. |
| `How does Citi define operational risk in its 2025 Form 10-K?` | Gubitak zbog neuspešnih internih procesa, ljudske/sistemske greške ili spoljnih događaja; uključuje cyber i legal risk, isključuje strategic/reputation risk. | Ručno verifikovan qrel + smoke. |
| `How does Goldman Sachs define cybersecurity risk in its 2025 Form 10-K?` | Kompromitovanje poverljivosti, integriteta ili dostupnosti podataka/sistema. | Ručno verifikovan qrel + smoke. |
| `How does JPMorgan Chase define cybersecurity risk in its 2025 Form 10-K?` | Rizik štete ili gubitka zbog zloupotrebe tehnologije ili neovlašćenog otkrivanja podataka. | Ručno verifikovan qrel + smoke. |
| `How does Capital One manage cybersecurity and technology risk under its enterprise risk framework?` | Operativni rizik, tri linije odbrane, lifecycle upravljanje i nadzor. | Stvarni UI QA, 3 izvora, audit passed. |

### Rezervna multi-bank pitanja

```text
How do Citi and JPMorgan Chase each define operational risk in their 2025 Form 10-K filings?
```

Očekivano: dve jasno odvojene definicije i bank-owned citati. Frozen evaluacija: deterministic,
semantic correctness, completeness i groundedness su prošli.

```text
What were the CET1 ratios for PNC and Truist Financial Corporation on December 31, 2025?
```

Očekivano: **PNC 10.6%**, **Truist 10.8%**. Frozen evaluacija je prošla.

### Namerno unsupported pitanje

```text
What was JPMorgan Chase & Co.'s Standardized CET1 capital ratio on December 31, 2026?
```

Očekivano: kontrolisan `unsupported` odgovor bez izmišljenog broja i bez lažnog citata, jer indeksirani
2025 Form 10-K ne podržava traženi datum 31. decembar 2026.

Ovo koristiti samo ako publika treba da vidi fail-closed ponašanje. Ne predstavljati `unsupported`
kao grešku sistema; to je ispravno ponašanje kada dokaz ne postoji.

---

## 8. Pitanja koja ne treba koristiti kao glavni demo

- Ne koristiti potpuno novo složeno pitanje koje nije u evaluacionim skupovima ako je cilj stabilan
  live nastup.
- Ne tražiti više od četiri banke: podržani comparison opseg je dve do četiri.
- Ne koristiti USB i WFC: nisu u aktivnom deset-bankovnom registry-ju, a stara dokumentacija beleži
  i nepotpune primarne filing priloge za njih.
- Ne oslanjati se na najnoviju cenu/vest ako web provider nije prethodno provereno dostupan.
- Ne koristiti `What is the latest stable Python version?` kao showcase routing-a zbog zabeleženog
  clarification promašaja.
- Ne menjati godinu, entity qualifier (`Corporation` naspram bank subsidiary) ili pristup
  (`Standardized` naspram `Advanced`) u numeričkom pitanju: to su deo činjenice, ne stil formulacije.
- Ne uploadovati poverljiv produkcioni dokument; koristiti deterministički fixture.
- Ne uključivati Agentic RAG neposredno pre demoa. Promena zahteva restart, a eksperimentalni rollout
  gate nije prihvaćen kao podrazumevani put.
- Ne otvarati `.env`, tokene, interne gateway URL-ove ili sirove logs pred publikom.

---

## 9. Šta pokazati u interfejsu

Kod svakog filing odgovora publika treba da vidi četiri različita sloja:

1. **Odgovor** — čitljiv Markdown rezultat.
2. **Citation chips** — koje evidence jedinice podržavaju tvrdnje.
3. **Source panel** — kanonski tekst ili kompletna tabela iz lokalnog korpusa.
4. **Diagnostics** — putanja izvršenja, broj dokaza, latencije, feature flag i ugovorne provere.

Kod comparison odgovora dodatno pokazati bank-specific kartice/sekcije. Kod web odgovora otvoriti
klikabilni HTTP(S) izvor. Kod upload odgovora naglasiti filename i document-owned citation.

### Kako objasniti status odgovora

| Status | Značenje |
|---|---|
| `supported` | Sve materijalne tvrdnje imaju validiran dokaz. |
| `partial` | Najmanje jedna banka/komponenta ima dokaz, ali druga nema dovoljno podrške. |
| `ambiguous` | Pitanje nije dovoljno precizno, npr. banka ili metric variant nisu jasni. |
| `unsupported` | Dostupan dokaz ne podržava traženu tvrdnju/period; sistem ne izmišlja. |

### Evidence audit

Runtime evidence audit je dodatna, savetodavna LLM provera finalnog filing odgovora samo nad citiranim
dokazom. Može biti `passed`, `review_recommended` ili `unavailable`. On ne menja deterministički
validiran odgovor i ne treba ga predstavljati kao formalnu eksternu reviziju.

---

## 10. Pre-demo checklist

### Dan ranije ili nekoliko sati ranije

- [ ] Ne menjati corpus, embeddings, Qdrant ili model posle uspešne probe.
- [ ] Potvrditi da `.env` sadrži važeći model gateway i Tavily fallback, bez prikazivanja vrednosti.
- [ ] Potvrditi da je `AGENTIC_RAG_ENABLED=false`.
- [ ] Potvrditi da `tests/fixtures/manual_upload_test.pdf` postoji.
- [ ] Zatvoriti druge Python procese koji mogu držati embedded Qdrant.
- [ ] Pripremiti ovaj Markdown i QA screenshot folder kao offline rezervu.

### Neposredno pre početka

Pokrenuti iz root-a:

```powershell
.\start-app.ps1
```

Sačekati poruku da je API spreman, pa u browseru otvoriti:

```text
http://localhost:5173
```

U drugom terminalu opciono proveriti:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

Očekivano:

```text
status
------
ready
```

Zatim:

- [ ] postaviti samo jedno kratko capability ili filing pitanje u test niti;
- [ ] proveriti da source panel može da se otvori;
- [ ] proveriti web prompt, jer je to jedini deo glavnog scenarija zavisan od spoljne mreže;
- [ ] uploadovati fixture u test nit i proveriti da se pojavljuje kao attachment;
- [ ] otvoriti novu, čistu nit za stvarni demo;
- [ ] uvećati browser na čitljivu veličinu i zatvoriti privatne tabove/notifikacije;
- [ ] ne pokretati evaluacije ili rebuild tokom prezentacije.

### Ako se API ne podigne

1. Proveriti da li prethodni BankScope Python proces još drži Qdrant.
2. Pročitati prvu konkretnu startup grešku; ne brisati indekse naslepo.
3. Potvrditi da `.venv\Scripts\python.exe` postoji.
4. Potvrditi da postoje `chunks.jsonl`, `tables.jsonl`, `embeddings.npz` i
   `qdrant_manifest.json`.
5. Ne pokretati `build_qdrant.py --recreate` pred demo osim ako je namerno urađen kompletan rebuild.

---

## 11. Plan B bez live modela ili interneta

Ako model gateway ili mreža zakažu, ne improvizovati. Otvoriti stvarne QA artefakte u:

```text
docs/final-report-assets/qa-demo-2026-08-31/
```

Preporučeni redosled:

1. `01-capital-one-cet1-answer.png`
2. `01-capital-one-cet1-source-evidence.png`
3. `01-capital-one-cet1-diagnostics.png`
4. `04-bac-citi-cet1-comparison-answer.png`
5. `04-bac-citi-cet1-comparison-source-evidence.png`
6. `04-bac-citi-cet1-comparison-diagnostics.png`

Za kvalitativni primer koristiti `02-capital-one-cyber-risk-*`, a za poređenje narativnih definicija
`03-citi-jpm-operational-risk-*`. `capture-results.json` sadrži tačna pitanja, stvarne odgovore,
dijagnostiku, vreme i lokalne thread URL-ove.

Rečenica za publiku:

> Spoljni model servis trenutno nije dostupan, zato pokazujem unapred reprodukovan stvarni prolaz
> kroz isti lokalni React i FastAPI sistem. Ovo nisu dizajnerske makete; sačuvani su odgovor,
> kanonski dokaz i dijagnostika izvršenja.

---

## 12. Teška pitanja publike i kratki odgovori

### „Zašto ovo nije samo ChatGPT nad PDF-om?”

Zato što BankScope poseduje domenski registry, verzionisani filing manifest, stabilne evidence ID-eve,
kanonske tabele, hibridni retrieval, bank ownership, stroge output šeme, proveru numeric tokena,
SQLite thread state i odvojene evaluacione gate-ove. LLM je jedna komponenta unutar kontrolisanog
cevovoda.

### „Kako sprečavate halucinacije?”

Ne postoji apsolutna zaštita, ali se rizik smanjuje fail-closed dizajnom: ograničen evidence payload,
obavezni poznati citati, exact numeric token u citiranom dokazu, entity/period/unit validacija,
kanonska hydration, bank izolacija i `unsupported` rezultat kada dokaz nije dovoljan.

### „Zašto ne čuvate celu tabelu direktno kao male chunk-ove?”

Zato što bi se vrednost mogla odvojiti od zaglavlja, perioda ili jedinice. Pretražuje se kompaktan
opis, ali se odgovor zasniva na celoj parser-emitted tabeli.

### „Zašto dva retrieval sistema?”

Dense je dobar za semantiku, BM25S za tačne termine i brojeve. RRF koristi prednosti oba bez
nepouzdanog direktnog poređenja njihovih skorova.

### „Da li prethodni odgovor postaje dokaz?”

Ne. Istorija je samo kontekst. Nova filing tvrdnja mora ponovo doći iz kanonskog korpusa. Samo
formatiranje/prevod prethodnog grounded odgovora može zadržati njegov dozvoljeni skup činjenica i
citata.

### „Šta ako jedna banka nema podatak?”

Ostale banke se i dalje nezavisno obrade. Rezultat je `partial`, uz eksplicitno objašnjenje koja
banka nema dovoljno dokaza.

### „Zašto je Agentic RAG isključen?”

Zato što još nije prošao potreban live additive-retrieval rollout gate. Pouzdanost i izmereni
baseline imaju prednost nad složenijim orchestration-om.

### „Da li sistem radi bez interneta?”

Lokalna pretraga nad već izgrađenim korpusom, Qdrant, BM25S, SQLite i UI su lokalni. Generisanje
odgovora trenutno zavisi od konfigurisanog OpenAI-compatible model gateway-a, a web ruta dodatno od
web providera. Offline retrieval se može pregledati kroz `scripts/search.py` bez LLM-a.

### „Da li može bilo koja banka?”

Ne automatski. Podržano je deset banaka iz `config/banks.yaml`. Nova banka zahteva registry zapis,
SEC download, corpus/embedding/index rebuild i evaluaciju.

### „Da li citat ostaje validan posle rebuild-a?”

Citation resolver proverava corpus hash. Zastareo ili nestao target se odbija, umesto da se otvori
možda pogrešan novi sadržaj.

### „Da li uploadovani dokument ide na web?”

Ne zbog samog upload-a. Čuva se lokalno u thread-scoped SQLite-u i koristi kao document-owned
dokaz. Model gateway dobija ograničen parsirani kontekst potreban za odgovor, u skladu sa trenutnim
lokalnim provider podešavanjem.

### „Šta je najveće trenutno ograničenje?”

Sistem je lokalni single-user prototip bez autentikacije i cloud deployment-a; generacija zavisi od
spoljnog model gateway-a; web zavisi od providera; Agentic RAG nije default; a svaka promena korpusa
ili retrieval arhitekture zahteva rebuild i ponovnu evaluaciju.

---

## 13. Korisne komande posle demoa

Pretraga bez generacije:

```powershell
python scripts/search.py "operational risk capital" --backend mixed --mode hybrid --ticker JPM
```

Jedan CLI odgovor:

```powershell
python scripts/answer.py "How does JPMorgan Chase define cybersecurity risk?"
```

Targetirani smoke:

```powershell
python scripts/smoke_qdrant.py
```

Aktivni Python quality gate:

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

Frontend gate:

```powershell
npm.cmd --prefix frontend run test
npm.cmd --prefix frontend run lint
npm.cmd --prefix frontend run build
```

Ne pokretati frozen live evaluacije ili full corpus/Qdrant rebuild samo radi demonstracije. Te
komande mogu biti spore, trošiti model pozive i promeniti lokalne generisane artefakte.

---

## 14. Završna poruka

> BankScope demonstrira da pouzdan RAG nije samo „pošalji pitanje i nekoliko chunk-ova LLM-u”.
> Pouzdanost dolazi iz čitavog lanca: kontrolisanog acquisition-a, stabilnih identiteta, lossless
> tabela, hibridnog retrieval-a, kanonske hydration, bankarske izolacije, stroge generacije,
> validacije citata i brojeva, bounded memorije, lokalne perzistencije i odvojenih evaluacija.
> Kada dokaz nije dovoljan, ispravan rezultat je da sistem to jasno kaže.

---

## 15. Izvori unutar repozitorijuma

- [`README.md`](../README.md) — sistemski pregled i arhitektura.
- [`scripts/README.md`](../scripts/README.md) — CLI komande.
- [`src/bankscope/README.md`](../src/bankscope/README.md) — Python package mapa i API.
- [`src/bankscope/generation/README.md`](../src/bankscope/generation/README.md) — routing, memory,
  generation i comparisons.
- [`src/bankscope/retrieval/README.md`](../src/bankscope/retrieval/README.md) — mixed retrieval i
  hydration.
- [`src/bankscope/parsing/README.md`](../src/bankscope/parsing/README.md) — `sec2md`, chunking i
  kanonske tabele.
- [`src/bankscope/chat/README.md`](../src/bankscope/chat/README.md) — SQLite, history, upload i
  citation source resolution.
- [`frontend/README.md`](../frontend/README.md) — React/Vite/SSE interfejs.
- [`data/README.md`](../data/README.md) — data lineage i artefaktni ugovori.
- [`data/evaluation/README.md`](../data/evaluation/README.md) — evaluacioni skupovi i rezultati.
- [`tests/fixtures/manual_upload_test_questions.md`](../tests/fixtures/manual_upload_test_questions.md)
  — deterministički upload odgovori.
- [`docs/final-report-assets/qa-demo-2026-08-31/README.md`](final-report-assets/qa-demo-2026-08-31/README.md)
  — stvarni UI QA scenariji i backup snimci.
- [`docs/decisions/013-rag-reliability-hardening.md`](decisions/013-rag-reliability-hardening.md)
  — trenutno prihvaćene reliability granice.
- [`docs/decisions/015-general-chat-web-and-calculator.md`](decisions/015-general-chat-web-and-calculator.md)
  — web, calculator i failure analysis.
- [`docs/decisions/016-finance-technology-conversation-scope.md`](decisions/016-finance-technology-conversation-scope.md)
  — aktivna finance/technology granica razgovora.
