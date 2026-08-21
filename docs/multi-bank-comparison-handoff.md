# Multi-bank comparison reliability — handoff za sledeći chat

## Cilj

Stabilizovati pitanja sa više tickera tako da svaka podržana banka dobije sopstveni relevantan
evidence paket, bank-specific generator ne očekuje dokaze druge banke, a završni comparison bude
`supported` kad su potrebne činjenice za sve banke zaista prisutne u indeksu.

Rešenje ne sme da hardkodira CET1 vrednosti, bankarske parove ili konkretne formulacije korisnika.
Model ostaje primarni semantički donosilac odluke, dok kod čuva bank isolation, evidence i citation
invarijante.

## Problem koji je primećen uživo

Upit:

```text
Compare Citigroup's CET1 ratio with JPMorgan's for 2025.
```

jednom je vratio `partial`: Citigroup je bio `unsupported`, a JPM `supported`, iako Citigroupov
2025 Form 10-K u indeksu sadrži odgovarajuću tabelu. Isti upit je u ponovljenom live testu uspeo:

- C: 13.18% Standardized i 11.93% Advanced, citat iz tabele;
- JPM: 14.6% Standardized i 14.1% Advanced, citat iz tabele;
- ukupni status: `supported`;
- retrieval: pet dokaza ukupno, dva za C i tri za JPM.

Ovo pokazuje da osnovna multi-bank putanja radi, ali da ostaje povremena nestabilnost u kvalitetu
bank-specific upita i odluci generatora da podrži ili odbaci pronađeni dokaz.

## Kako je multi-bank ranije radio

### Prva implementacija — ADR 008

Commit linija: `16772a2` i kasnije izmene comparison synthesisa.

- Dve do četiri banke formirale su ordered comparison scope.
- Jedno standalone comparison pitanje se embeddingovalo jednom.
- Retrieval se izvršavao zasebno za svaki ticker.
- Svaki ticker imao je izolovan evidence paket i zaseban generation poziv.
- Validirani bankarski odgovori dobijali su globalne `E1...En` oznake.
- Poseban synthesis poziv je dobijao validirane bank results, ne sirove filing dokaze.
- Jedna neuspešna banka davala je `partial`; više od četiri banke odbijano je pre retrievala.

Frozen comparison gate tada je prošao 3/3 pitanja i 6/6 evidence grupa bez cross-ticker citation
prekršaja. Detalji su u `docs/decisions/008-multi-bank-comparisons.md`.

### Bank-balanced retrieval — ADR 010

Globalno spajanje rezultata dozvoljavalo je jednoj banci da zauzme većinu Top 10. Zato je uvedeno:

- jedan mixed-hybrid search po banci;
- `limit=5`, `candidate_k=30`, `rrf_k=60` po tickeru;
- izolovani per-bank result paketi za generation;
- deterministički round-robin samo za zbirne retrieval metrike.

Ovaj pristup je prošao sva tri tadašnja cross-bank pitanja (`BAC/C`, `C/JPM`, `PNC/TFC`) i svih
6/6 required evidence grupa. Detalji su u `docs/decisions/010-bank-balanced-comparison-retrieval.md`.

### Peer-free podupiti — ADR 013

Kasnije je zajednički comparison tekst prepoznat kao izvor razvodnjavanja retrievala. Uveden je
`build_bank_subquestion()` koji uklanja peer banke i pravi zaseban upit za svaki ticker. Svaki od tih
upita se nezavisno embeddinguje i pretražuje. Ovo je sadašnja dobra osnova i treba je zadržati.

## Šta se promenilo

### 1. Model-first router sada generiše bogatiji standalone question

Router može da vrati tekst poput:

```text
Compare Citigroup Inc. (ticker: C) CET1 ratio with JPMorgan Chase & Co. (ticker: JPM) ...
```

`build_bank_subquestion()` uklanja nazive i aliase banaka, ali ne uklanja strukturne ticker oznake.
Live dijagnostika zato pokazuje upite sa šumom:

```text
Citigroup Inc. (C) Form 10-K: ticker c cet1 ratio ticker cet1 ratio for 2025 ...
JPMorgan Chase & Co. (JPM) Form 10-K: ticker c cet1 ratio ticker cet1 ratio for 2025 ...
```

Rezervni queries (`cet1 ratio for 2025` i `CET1 common equity tier 1 capital ratio`) često ipak
pronađu pravu tabelu, ali prvi upit je lošiji i može doprineti nestabilnom rangiranju.

### 2. Bank-specific generator je do poslednje izmene video dva različita scope-a

Pipeline je pravio dobar `resolved_question=bank_question`, ali je kao `Current user question`
generatoru slao originalno poređenje sa obe banke. Model je zato povremeno zaključivao da izolovani
evidence paket nije dovoljan za ceo zahtev i vraćao `unsupported` za jednu banku.

U trenutnom working tree-u ovo je popravljeno: comparison grana sada prosleđuje
`original_bank_question` kao current question i `bank_question` kao resolved question. JPM generator
više ne vidi BAC/C, i obrnuto. Regression test eksplicitno proverava ovu izolaciju.

Posle ove izmene:

- JPM/BAC live poređenje je bilo potpuno `supported` (14.6% naspram 11.4%);
- C/JPM live ponavljanje je bilo potpuno `supported`;
- ceo backend suite je prošao: 295 testova.

Važno: izmene su trenutno u prljavom, necommitovanom working tree-u zajedno sa širim model-first
memory radom. Ne resetovati ili checkoutovati ove fajlove.

### 3. Presentation guidance je dodat u sve generation slojeve

Router sada prosleđuje stilsku instrukciju i bank-specific generatorima i synthesis-u. To nije glavni
uzrok `unsupported` rezultata, ali promptovi imaju više instrukcija nego ranije i treba evaluatorom
proveriti da stil nikada ne utiče na evidence odluku ili coverage svih banaka.

### 4. Strict function calling je zamenio plain JSON synthesis

Comparison synthesis sada koristi strict tool poziv. Ovo poboljšava izlaznu šemu i citate. Ne treba
ga vraćati na plain JSON; neuspeh koji analiziramo nastaje pre synthesisa, u pojedinačnom bank resultu.

## Predloženo rešenje

### Faza 1 — očistiti bank-specific query

Pre uklanjanja naziva banaka normalizovati samo strukturne ticker anotacije:

```text
(ticker: C)
ticker C
(C)
(ticker: JPM)
ticker JPM
(JPM)
```

Ovo mora biti vezano za tickere iz već deterministički razrešenog `selected_tickers` skupa. Ne
uklanjati proizvoljno slovo `C` iz prirodnog teksta. Nakon čišćenja, svaki podupit treba da izgleda
otprilike ovako:

```text
Citigroup Inc. (C) Form 10-K: cet1 ratio for 2025
JPMorgan Chase & Co. (JPM) Form 10-K: cet1 ratio for 2025
```

Dodati unit testove za model-generated `(ticker: C)` i `(ticker: JPM)` oblik, possessive nazive i
obrnut redosled banaka.

### Faza 2 — evidence-aware generation retry

Ako pojedinačni bank generator vrati `unsupported`, ne pretvarati ga automatski u supported. Prvo
lokalno utvrditi da li evidence paket ima jake signale za traženi odgovor:

- traženu godinu ili kompatibilan reporting period;
- naziv metrike ili pouzdan sinonim;
- tabelarni zapis ili tekst sa numeričkom vrednošću;
- ispravan ticker i record scope.

Ako su signali prisutni, dozvoliti tačno jedan dodatni model poziv sa bank-only instrukcijom:

```text
Re-evaluate only the expected bank. Do not require evidence for peer banks. Inspect the supplied
table/text for the requested metric and period. Use only supplied evidence and abstain if it still
does not directly support the answer.
```

Retry mora ostati fail-closed: nema lokalnog izvlačenja i objavljivanja broja bez validiranog model
odgovora i citata. U dijagnostici beležiti `bank_generation_retry`, razlog i request count.

### Faza 3 — ciljani drugi retrieval prolaz

Ako prvi evidence paket nema jake metric/period signale, pokrenuti drugi retrieval samo za neuspešnu
banku. Ne ponavljati uspešne banke. Upit proširiti kontrolisanim terminima izvedenim iz metrike, na
primer za CET1:

```text
CET1
Common Equity Tier 1
CET1 capital ratio
Standardized Approach
Advanced Approaches
Regulatory Capital
December 31 2025
```

Poželjno je da sinonime predloži postojeći query-planning sloj ili model sa strict izlazom, ali kod
mora da zadrži originalnu metriku, godinu i ticker. Novi dokazi se dodaju postojećim, deduplikuju i
ponovo rangiraju; prvi validni evidence se ne sme izgubiti.

Drugi retrieval se pokreće samo kada je opravdan nedostatkom dokaza, kako se ne bi povećali latency i
trošak svakog poređenja.

### Faza 4 — evaluacija stabilnosti, ne samo jednog prolaza

Postojeći testovi dobro proveravaju strukturu, ali mock model uvek daje isti odgovor. Dodati live ili
snimljeni no-regression benchmark koji uključuje:

- `C vs JPM`, `JPM vs C`;
- `BAC vs JPM`, `JPM vs BAC`;
- `C vs BAC`, `BAC vs C`;
- tri banke u jednom pitanju;
- Standardized i Advanced pristupe;
- explicit 2025 i follow-up period;
- najmanje tri ponavljanja svakog kritičnog pitanja.

Prihvatni kriterijumi:

- 100% očekivanih ticker-a ima bank result;
- 100% ručno potvrđenih evidence grupa je pronađeno;
- nula cross-ticker citation prekršaja;
- nema `partial` ako svaki per-bank evidence paket sadrži ručno potvrđen odgovor;
- `partial` ostaje dozvoljen kada dokaz za jednu banku zaista nedostaje;
- request budget ostaje ograničen i vidljiv u dijagnostici.

## Preporučeni redosled rada

1. Sačuvati trenutni working tree i pokrenuti postojeće testove pre nove izmene.
2. Dodati ticker-annotation cleanup u `build_bank_subquestion()` i unit testove.
3. Pokrenuti frozen retrieval/comparison evaluatore iz postojećih skripti.
4. Dodati evidence-strength proveru i jedan bank-specific generation retry.
5. Tek zatim dodati ciljani drugi retrieval prolaz za stvarne evidence miss-eve.
6. Ponoviti kritična live pitanja više puta i uporediti rezultate sa baseline-om.
7. Ažurirati ADR 013 ili dodati novi ADR sa izmerenim rezultatima.

## Relevantni fajlovi

- `src/bankscope/generation/pipeline.py` — comparison orchestration i per-bank generation.
- `src/bankscope/generation/query_planner.py` — `build_bank_subquestion()` i retrieval queries.
- `src/bankscope/generation/answer_generator.py` — grounded bank answer i retry politika.
- `src/bankscope/generation/comparison_generator.py` — finalni validated synthesis.
- `tests/test_answer_pipeline.py` — comparison isolation i partial behavior.
- `tests/test_query_planner.py` — peer-free podupiti.
- `scripts/evaluate_comparisons.py` — frozen comparison gate.
- `docs/decisions/008-multi-bank-comparisons.md` — originalni comparison dizajn.
- `docs/decisions/010-bank-balanced-comparison-retrieval.md` — per-bank retrieval kvote.
- `docs/decisions/013-rag-reliability-hardening.md` — peer-free query dizajn.

## Ne raditi

- Ne hardkodirati vrednosti 13.18%, 11.93%, 14.6% ili 14.1%.
- Ne uvoditi posebnu granu samo za `C`, JPM ili CET1 korisničku rečenicu.
- Ne dozvoliti da dokaz jedne banke podrži tvrdnju druge banke.
- Ne pokretati synthesis nad sirovim filing evidence-om; koristiti samo validirane bank results.
- Ne pretvarati `unsupported` u `supported` samo zato što evidence izgleda obećavajuće.
- Ne vraćati globalni cross-bank Top-K koji može da izgladni jednu banku.
- Ne resetovati postojeći prljavi working tree.

## Kratka početna poruka za sledeći chat

```text
Pročitaj docs/multi-bank-comparison-handoff.md, posebno završni status od 2026-08-21. Faze 1–4 su
implementirane i automated gate-ovi prolaze. Restartuj backend i pomozi mi da ručno proverim C/JPM,
BAC/JPM i trostruka poređenja. Ne hardkodiraj banke ili CET1 vrednosti i ne menjaj bank
isolation/citation invarijante bez nove evaluacije.
```

## Implementirano i evaluirano 2026-08-21

Faze 1–4 iz ovog dokumenta su implementirane:

- `build_bank_subquestion()` bezbedno uklanja samo strukturne anotacije za već razrešene tickere i
  sažima neposredno duplirane topic fraze;
- prepoznata fokusirana metrika dobija jedan bank-scoped drugi retrieval samo kada prvi evidence
  paket nema potrebne concept/period/numeric signale;
- jak evidence paket dobija najviše jedan bank-only generation recheck posle model abstention-a;
- post-validation `invalid_citations` dobija isti ograničeni recheck, ali `invalid_schema` ostaje
  fail-closed;
- numeric odgovor sa nepoznatom citation labelom može se lokalno uskladiti samo sa evidence labelama
  koje doslovno sadrže isti `facts.value_text`; bez exact-value dokaza odgovor se odbija;
- comparison evaluator podržava `--repetitions` i ispravno razlikuje preskočen semantic judge od
  neuspešnog semantic gate-a.

Tokom evaluacije je potvrđeno da Truist Table 37 target
`d46c08d204efc3fe045283f309ca5f409886b2f73f31726d80e4d8e3686832c0` direktno navodi
Corporation CET1 od 10.8% za 2025. Target je već bio ručno prihvaćen u
`generation_citation_audit_v2.jsonl`, pa je dodat kao dozvoljena alternativa odgovarajućoj frozen
cross-bank evidence grupi.

Rezultati:

- kompletan backend suite: **302 passed**;
- Ruff: **all checks passed**;
- mixed-hybrid frozen retrieval: Hit@5 **31/32**, Hit@10 **32/32**, cross-bank **3/3 pitanja i 6/6
  grupa**, bez Top-5 ili Top-10 regresija;
- finalni comparison stability run (`--skip-judge --repetitions 3`): **9/9 runs**, **3/3 stable
  queries**, **18/18 evidence groups**, **0 citation ownership violations**, deterministic i overall
  gate **pass**;
- semantic judge nije pokrenut u finalnom stability run-u i zato je zabeležen kao `null`, ne kao
  neuspeh.

Generisani lokalni izveštaji:

- `data/evaluation/results/codex-multibank-retrieval.json`;
- `data/evaluation/results/codex-multi-bank-comparison.json`;
- `data/evaluation/results/codex-multi-bank-comparison-stability.json`.

Za ručni UI test potreban je restart backend-a, jer trenutno pokrenuti proces ne učitava ove poslednje
izmene dok se ne restartuje.
