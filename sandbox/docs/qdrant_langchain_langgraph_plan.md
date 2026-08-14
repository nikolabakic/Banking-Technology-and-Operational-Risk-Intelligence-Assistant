# Plan nastavka: Qdrant, LangChain i LangGraph

> **Archived:** This pre-implementation proposal is retained only as historical context. It does
> not describe the active architecture. Use the root README, current roadmap, and ADRs 003-004.

## 1. Svrha dokumenta

Ovaj dokument predlaže narednu fazu BankScope projekta nakon izgradnje
whole-table corpusa i GPU evaluacije Qwen embeddinga. Plan polazi od sledećeg
željenog tehnološkog pravca:

- Qdrant Local kao lokalna vector baza;
- Qdrant dense i sparse pretraga;
- Reciprocal Rank Fusion (RRF) unutar Qdrant query toka;
- LangChain kao integracioni sloj za model, alate i retrieval;
- LangGraph kao eksplicitna orkestracija razgovora i poziva alata;
- Pydantic samo tamo gde daje jasan ugovor i validaciju.

Ovo je radni arhitektonski plan, a ne odluka da se svaki navedeni alat mora
zadržati bez obzira na rezultate. Svaka promena retrieval metode mora proći
postojeću evaluaciju pre nego što zameni aktivni baseline.

UI, Streamlit, javni deployment i produkciona infrastruktura nisu deo ove
faze.

## 2. Potvrđeno početno stanje

Plan se oslanja samo na sledeće proverene činjenice iz aktivnog projekta:

- corpus ima 5.565 retrieval zapisa;
- postoji 4.009 tekstualnih i 1.556 table retrieval zapisa;
- `tables.jsonl` sadrži 1.783 cele parser-emitted tabele;
- table retrieval pretražuje opis, a zatim vraća celu tabelu kao dokaz;
- dense embedding koristi `Qwen/Qwen3-Embedding-0.6B`;
- embedding matrica ima oblik `(5565, 1024)` i normalizovane `float32` vektore;
- `chunks.jsonl`, redosled embeddinga i `embeddings.npz` povezani su hash i ID
  ugovorom;
- trenutni lexical baseline koristi `bm25s` sa projektno definisanom
  tokenizacijom;
- trenutni hybrid koristi RRF sa `candidate_k=30` i `rrf_k=60`;
- GPU rezultat je izračunat na 28 odgovorivih pitanja iz skupa od 30 pitanja.

Trenutni rezultat koji nova implementacija mora eksplicitno da poredi je:

| Metoda | Hit@1 | Hit@5 | Hit@10 | MRR@10 | Mean Recall@10 |
|---|---:|---:|---:|---:|---:|
| Dense | 10/28 | 22/28 | 23/28 | 0,522 | 0,669 |
| BM25S | 11/28 | 25/28 | 26/28 | 0,562 | 0,787 |
| Hybrid RRF | 11/28 | 25/28 | 26/28 | 0,614 | 0,855 |

Izvor detaljnog lokalnog rezultata je
`data/evaluation/results/retrieval.json`, a provenance je u
`data/evaluation/results/run_provenance.json`.

## 3. Cilj naredne faze

Naredna faza treba da proizvede lokalni retrieval i orchestration sloj koji:

1. učitava postojeće dense embeddinge u Qdrant bez ponovnog dense embedovanja;
2. generiše izabranu sparse reprezentaciju iz postojećeg `embedding_text`;
3. izvršava dense, sparse i Qdrant-native RRF pretragu;
4. filtrira rezultate po proverljivom metadata payloadu;
5. posle table hita vraća originalnu celu tabelu;
6. može da se izloži kao LangChain alat;
7. može da se poziva iz malog, eksplicitnog LangGraph workflowa;
8. zadržava citate i source provenance kroz ceo tok;
9. odbija zastarelu ili nekompatibilnu lokalnu bazu;
10. meri retrieval i answer kvalitet odvojeno.

Uspeh ove faze ne znači samo da Qdrant vraća rezultate. Uspeh znači da je
novi rezultat najmanje jednako pouzdan kao prihvaćeni baseline ili da je svaka
regresija svesno prihvaćena zbog jasno izmerene koristi.

## 4. Predložena arhitektura

```text
User question
    |
    v
LangGraph state and routing
    |
    v
LangChain retrieval tool
    |
    v
BankScope retrieval service
    |
    +--> Qwen query encoder
    |
    +--> sparse query encoder
    |
    +--> Qdrant Local
           - dense named vector
           - sparse named vector
           - metadata payload
           - native RRF query
    |
    v
Evidence hydration
    - narrative chunk from retrieval record
    - complete table from tables.jsonl
    |
    v
LLM answer with citation IDs
    |
    v
Deterministic citation validation
```

Arhitektura namerno zadržava BankScope retrieval servis između frameworka i
baze. Poslovna pravila projekta ne treba direktno vezati za LangChain
`Document` niti za sirov Qdrant response objekat.

## 5. Predložena struktura foldera

Struktura je predlog i treba je potvrditi pre implementacije:

```text
config/
  banks.yaml                       postojeći bank registry
  retrieval.yaml                   Qdrant i retrieval parametri

data/
  processed/
    chunks.jsonl                   izvor retrieval zapisa
    tables.jsonl                   izvor celih tabela
    manifest.json                  corpus provenance
    embeddings.npz                 postojeći dense vektori
    qdrant/                        generisana lokalna baza, Git ignored
    qdrant_manifest.json           ugovor baze sa izvorima, Git ignored
  evaluation/
    queries.jsonl                  postojeći retrieval qrels
    results/
      retrieval.json               prihvaćeni rezultat evaluacije

src/bankscope/
  vectorstore/
    __init__.py
    qdrant_store.py                collection lifecycle i Qdrant upiti
    point_mapping.py               corpus record -> Qdrant point
    sparse_encoder.py              izabrani sparse encoder
    manifest.py                    hash/model/schema ugovor baze
  retrieval/
    hybrid_retriever.py            postojeći baseline tokom migracije
    service.py                     stabilni BankScope retrieval interfejs
    evidence.py                    table hydration i citation identiteti
  assistant/
    schemas.py                     mali broj spoljašnjih data ugovora
    tools.py                       LangChain tool adapteri
    state.py                       LangGraph state
    graph.py                       graph nodes, edges i stop pravila
    prompts.py                     retrieval i answer pravila
  evaluation/
    retrieval_metrics.py           postojeće metrike
    answer_metrics.py              kasnija generation evaluacija

scripts/
  index_qdrant.py                  izgradi/proveri lokalnu kolekciju
  search.py                        postojeći CLI, dobija izbor backenda
  evaluate.py                      postojeći evaluator, dobija izbor backenda
  assistant.py                     kasniji CLI bez UI-ja

tests/
  test_qdrant_mapping.py
  test_qdrant_manifest.py
  test_qdrant_retrieval.py
  test_sparse_encoder.py
  test_evidence_hydration.py
  test_assistant_schemas.py
  test_assistant_graph.py
```

Ne treba unapred praviti sve fajlove iz stabla. Folderi i moduli se dodaju po
fazama, kada za njih postoji konkretna odgovornost i test.

## 6. Qdrant collection ugovor

### 6.1 Collection

Radni naziv kolekcije može biti `bankscope_retrieval_v1`. Konačan naziv ostaje
konfigurabilan jer nova corpus ili schema verzija može zahtevati novu
kolekciju.

Kolekcija treba da ima dva named vector polja:

- `dense`: 1.024 dimenzije, cosine distance;
- `sparse`: sparse indices i weights iz izabranog lexical encodera.

RRF spaja rangove dense i sparse pretrage. Početni Qdrant RRF parametri treba
da odgovaraju postojećem evaluatoru gde god Qdrant API to omogućava. Razlike u
semantici parametara moraju biti dokumentovane, ne pretpostavljene.

### 6.2 Point ID

Postojeći `record_id` nije unapred pretpostavljen kao validan Qdrant point ID.
Predlog je deterministički UUID izveden iz `record_id`, dok originalni
`record_id` ostaje u payloadu.

Mapiranje mora biti:

- determinističko;
- bez kolizija u aktivnom corpus-u;
- proverljivo pri ponovnom indeksiranju;
- nezavisno od redosleda batch upload operacija.

### 6.3 Payload

Minimalni payload treba da obuhvati:

- `record_id`;
- `target_chunk_id`;
- `record_type`;
- `ticker`;
- `table_id`, kada postoji;
- `accession_number`;
- filing/report datum ili godinu, ako su pouzdano prisutni u corpus metadata;
- page i section provenance, ako su prisutni;
- `embedding_text`;
- narrative `document`, ili drugi dovoljan ključ za hydration.

Ne treba izmišljati metadata polja koja corpus trenutno nema. Point mapper
mora eksplicitno razlikovati obavezna, opcionalna i zabranjena polja.

Cela Markdown tabela ne treba da se kopira u svaki table retrieval point.
`table_id` ostaje veza prema `tables.jsonl`, koji je izvor dokaza.

### 6.4 Manifest lokalne baze

Qdrant folder sam po sebi nije dovoljan provenance ugovor. Pored baze treba
čuvati `qdrant_manifest.json` sa najmanje:

- schema version;
- collection name;
- point count;
- SHA-256 za `chunks.jsonl` i `tables.jsonl`;
- dense model name i revision;
- dense embedding archive SHA-256;
- sparse model/algorithm name i revision, kada postoji;
- tokenizer ili encoder konfiguraciju koja utiče na sparse rezultate;
- fusion metod i relevantne parametre;
- vreme izgradnje;
- verzije ključnih biblioteka.

Search i evaluation moraju odbiti kolekciju ako manifest ne odgovara aktivnom
corpus-u. Ne sme postojati tihi fallback na zastarelu bazu.

## 7. Dense migracija

Dense migracija treba da koristi postojeći `embeddings.npz`. Ponovno kodiranje
svih 5.565 zapisa nije potrebno osim ako se promeni model, input ili embedding
ugovor.

Koraci su:

1. učitati `chunks.jsonl` i proveriti jedinstvene `record_id` vrednosti;
2. učitati NPZ kroz postojeći validacioni loader;
3. proveriti input hash, model revision, broj i redosled ID-jeva;
4. mapirati svaki corpus zapis i pripadajući vektor u Qdrant point;
5. uploadovati u determinističkim batch-evima;
6. proveriti collection count i uzorak point/payload mapiranja;
7. sačuvati manifest;
8. izvršiti dense parity evaluaciju.

Prva dense evaluacija treba da koristi exact search, ako izabrani Qdrant Local
put to pouzdano podržava. Na ovom corpus-u approximate search nije potreban da
bi baza stala u memoriju. HNSW ili drugi approximate režim treba razmatrati tek
nakon exact parity rezultata i merenja latencije.

## 8. Sparse pretraga: otvorena odluka

Termin "Qdrant sparse" ne određuje automatski kako nastaju sparse weights. To
je najvažnija otvorena tehnička odluka ove faze.

Treba uporediti najmanje sledeće:

### Varijanta A: Qdrant/FastEmbed BM25 sparse encoder

Prednosti:

- prirodno se uklapa u Qdrant sparse i LangChain Qdrant integraciju;
- lokalno generisanje;
- jednostavniji single-database hybrid tok;
- objašnjiv lexical signal.

Rizik:

- tokenizacija i scoring nisu garantovano isti kao postojeći `bm25s` baseline;
- postojeći rezultat zato ne može samo da se prenese na novu varijantu.

Ovo je preporučeni prvi Qdrant-native sparse eksperiment.

### Varijanta B: learned sparse encoder, na primer SPLADE porodica

Prednosti:

- može proširiti termine i pomoći kod sinonima i aliasa;
- može poboljšati lexical-semantic recall.

Rizici:

- dodatni model i veće vreme indeksiranja/upita;
- složenije objašnjenje i reprodukcija;
- nije opravdan bez merljivog dobitka na našim pitanjima.

Ovu varijantu ne treba uvoditi pre BM25 sparse baseline-a.

### Kontrolna varijanta: postojeći BM25S

Postojeći `bm25s` ostaje kontrola tokom migracije. Ne mora ostati u konačnoj
arhitekturi, ali se ne uklanja dok Qdrant-native sparse/hybrid ne prođe
evaluacionu kapiju.

## 9. RRF i retrieval eksperimenti

Minimalna matrica eksperimenata je:

| ID | Dense | Sparse | Fusion | Svrha |
|---|---|---|---|---|
| E0 | NumPy | BM25S | postojeći RRF | prihvaćeni baseline |
| E1 | Qdrant exact | bez sparse | bez fusiona | dense parity |
| E2 | bez dense | Qdrant BM25 sparse | bez fusiona | sparse kvalitet |
| E3 | Qdrant exact | Qdrant BM25 sparse | Qdrant RRF | ciljni local hybrid |
| E4 | Qdrant approximate | Qdrant BM25 sparse | Qdrant RRF | opcioni speed test |
| E5 | Qdrant | learned sparse | Qdrant RRF | samo ako E3 nije dovoljan |

Weighted RRF ne treba podešavati na svih 28 pitanja i zatim prijaviti rezultat
na istim pitanjima kao nepristrasnu evaluaciju. Sa ovako malim skupom možemo:

- zadržati običan RRF kao glavni rezultat; ili
- unapred definisati mali development/validation split i jasno prijaviti veliku
  statističku neizvesnost.

Svaki eksperiment treba da sačuva model, revision, corpus hash, sparse
konfiguraciju, fusion parametre, filtere i per-query rangove.

## 10. Evaluacione kapije

### Gate 1: integritet indeksa

- 5.565 pointova;
- svaki `record_id` postoji tačno jednom;
- svaki `target_chunk_id` odgovara aktivnom corpus-u;
- dense vektor ima 1.024 dimenzije;
- svi source hash-evi se poklapaju;
- svi table pointovi mogu da se hydrate-uju;
- filteri ne vraćaju pogrešnu banku ili tip zapisa.

### Gate 2: dense parity

Kod exact cosine pretrage očekuje se isti ili funkcionalno ekvivalentan dense
ranking. Eventualne razlike zbog tie-break pravila moraju biti izdvojene od
stvarnih promena relevantnosti.

Minimum za prihvatanje je:

- nema gubitka qrel pokrivenosti izazvanog pogrešnim point mapiranjem;
- agregatne dense metrike nisu lošije od prihvaćenog dense rezultata;
- razlike u top rezultatima imaju objašnjiv uzrok.

### Gate 3: Qdrant sparse/hybrid kvalitet

Ciljni Qdrant-native hybrid upoređuje se sa:

- BM25S: Hit@5 25/28, Hit@10 26/28, MRR@10 0,562;
- postojeći hybrid: Hit@5 25/28, Hit@10 26/28, MRR@10 0,614;
- postojećim per-query i cross-bank evidence-group rezultatima.

Nijedna pojedinačna metrika nije dovoljna. Posebno se pregledaju:

- alias/expansion pitanja;
- table exact-value pitanja;
- split-table slučajevi;
- cross-bank kompletna pokrivenost;
- pitanja koja su prethodno padala na @10.

### Gate 4: latencija

Ne treba unapred tvrditi da je Qdrant brži. Merimo odvojeno:

- cold application start;
- učitavanje dense i sparse query encodera;
- query embedding vreme;
- Qdrant dense search vreme;
- Qdrant sparse search vreme;
- Qdrant fusion vreme;
- hydration vreme;
- ukupan retrieval bez LLM-a.

Merenje treba ponoviti na istoj mašini, sa warm i cold scenarijem. OpenAI API
latencija ne ulazi u poređenje vector backenda.

## 11. Uloga LangChaina

LangChain u ovoj arhitekturi treba da bude integracioni sloj, ne vlasnik
corpus-a niti evaluacione logike.

Planirana upotreba:

- LangChain OpenAI model adapter;
- LangChain tool definicija za BankScope retrieval;
- standardni message objekti;
- eventualno Qdrant vector-store adapter, ako ne ograničava native query i
  provenance zahteve;
- testabilan tool input/output ugovor;
- lakše povezivanje sa LangGraph ToolNode mehanizmom.

Neplanirana upotreba u prvoj verziji:

- ponovno učitavanje ili chunkovanje SEC dokumenata kroz LangChain loadere;
- LangChain text splitter umesto postojećeg structure-aware corpusa;
- automatsko re-embedovanje postojećih zapisa;
- generički retrieval chain koji skriva table hydration;
- menjanje qrel identiteta u LangChain-generated ID-jeve.

Ako `langchain-qdrant` ne izlaže Qdrant-native sparse/RRF kontrolu potrebnu za
evaluaciju, core retrieval treba da koristi zvanični Qdrant client direktno.
LangChain alat tada samo poziva BankScope servis. Korišćenje LangChaina nije
važnije od očuvanja retrieval ugovora.

## 12. Uloga LangGrapha

LangGraph treba uvesti tek nakon što Qdrant retrieval servis samostalno prođe
evaluaciju. Graph ne treba koristiti za indexing pipeline.

Početni graph treba da ostane mali:

```text
START
  |
  v
assistant/model node
  |-- no tool call ----------------------> validate answer --> END
  |
  +-- one or more retrieval tool calls --> ToolNode
                                             |
                                             v
                                      hydrate/validate evidence
                                             |
                                             v
                                      assistant/model node
```

Model može napraviti više retrieval tool poziva za cross-bank ili multi-part
pitanja. Aplikacija mora imati eksplicitni maksimalni broj iteracija i tool
poziva da bi se sprečile beskonačne petlje i nepotrebni API troškovi.

Predloženi minimalni state sadrži:

- `messages`;
- prikupljene `evidence` objekte;
- dozvoljene citation ID-jeve;
- status odgovora;
- broj retrieval iteracija;
- eventualnu dijagnostiku, ali ne sirove interne reasoning tragove.

Conversation persistence nije obavezna u prvoj graph iteraciji. Može se dodati
kasnije kroz LangGraph checkpointer kada definišemo zahteve za istoriju i
privatnost.

## 13. Da li je Pydantic potreban?

LangChain i LangGraph koriste tipove i Pydantic na delovima svojih interfejsa,
ali time ne definišu automatski BankScope ugovore. Pydantic je već direktna
zavisnost projekta i treba ga koristiti selektivno.

Predložena eksplicitna upotreba:

- `RetrievalRequest`;
- `Evidence`;
- `Citation`;
- `AssistantAnswer`;
- učitavanje i validacija konfiguracije kroz `pydantic-settings`, gde ima
  smisla;
- tool input schema koja se prosleđuje LangChainu/modelu.

LangGraph interni state može ostati `TypedDict`, jer nije potrebno da se svaki
graph update rekonstruiše kao Pydantic model. Pydantic koristimo na granicama:
korisnički/tool input, database output, finalni odgovor i konfiguracija.

Ovo izbegava i dupliranje modela i oslanjanje na neproverene rečnike između
komponenti.

## 14. Retrieval tool ugovor

Prva verzija treba da ima jedan glavni retrieval alat. Radni naziv može biti
`search_bank_filings`.

Ulazna polja treba ograničiti na podatke koje aplikacija zaista može da
primeni, na primer:

- `query`;
- `ticker` ili `null`;
- `record_type`: `text`, `table`, `any`;
- `year` samo ako corpus metadata omogućava pouzdan filter;
- `limit` u malom dozvoljenom opsegu.

Model ne treba da bira Qdrant collection, embedding model, sparse model,
fusion metod ili `rrf_k`. To su evaluirani parametri aplikacije, ne deo
korisničke namere.

Rezultat alata treba da sadrži:

- stabilni `evidence_id` izveden iz `target_chunk_id`;
- banku i filing provenance;
- record type;
- sekciju/page podatke kada postoje;
- hydrated evidence tekst;
- retrieval dijagnostiku samo ako je uključena u development režimu.

Finalni model sme da citira samo `evidence_id` vrednosti koje su vraćene u
tekućem graph run-u. Aplikacija to proverava deterministički.

## 15. Query decomposition i aliasi

Qdrant sparse retrieval i learned sparse modeli mogu pomoći terminološkim
varijantama, ali ne treba pretpostaviti da rešavaju sve aliase.

Redosled je:

1. izmeriti Qdrant BM25 sparse na postojećim alias pitanjima;
2. proveriti da li model ili sparse encoder već rešava problem;
3. tek zatim dodati mali provereni alias registry ako je potreban;
4. query decomposition koristiti za pitanja koja zahtevaju više nezavisnih
   dokaza, naročito više banaka;
5. meriti rezultat pre i posle promene.

LangGraph omogućava paralelne ili ponovljene tool pozive, ali odluka da se
pitanje razloži mora ostati vidljiva u test rezultatima. Ne treba praviti
poseban autonomni agent samo za decomposition bez dokazane potrebe.

## 16. Test strategija

### Unit testovi

- corpus record uvek daje isti Qdrant point ID;
- payload odbija nedostajuća obavezna polja;
- manifest odbija pogrešan corpus hash;
- sparse encoder vraća validne, konačne i deterministički uređene sparse
  indices/values;
- table evidence se hydrate-uje iz tačnog `table_id`;
- ticker i record-type filteri ne propuštaju pogrešne rezultate;
- citation validator odbija nepostojeći evidence ID;
- graph se zaustavlja na definisanom iteration limitu.

### Integracioni testovi

- privremena Qdrant Local baza u test folderu;
- mali corpus sa text i table zapisom;
- dense-only, sparse-only i hybrid upit;
- filteri;
- restart klijenta i ponovno otvaranje persistent baze;
- LangChain tool poziva BankScope servis;
- LangGraph ToolNode vraća rezultat model node-u uz mock model.

### Evaluacioni testovi

- svih 28 odgovorivih qrels pitanja;
- ista `DEFAULT_K_VALUES` i evidence-group pravila;
- per-query diff naspram baseline-a;
- rezultat se čuva sa potpunim provenance podacima;
- OpenAI pozivi nisu deo retrieval evaluacije.

Test suite po defaultu ne treba da zahteva OpenAI key, internet niti ponovno
preuzimanje modela. Realni LLM integracioni test treba da bude eksplicitno
opt-in.

## 17. Dependency plan

Tačne verzije ne treba pretpostaviti pre compatibility probe-a sa Python 3.13.
Kandidati za novu optional dependency grupu su:

- `qdrant-client` sa potrebnom local/fastembed podrškom;
- `langchain`;
- `langchain-openai`;
- `langchain-qdrant`;
- `langgraph`;
- postojeći `pydantic` i `pydantic-settings`;
- postojeći `sentence-transformers` za Qwen query embedding.

Pre izmene `pyproject.toml` treba proveriti:

- Python 3.13 podršku svake verzije;
- kompatibilnost LangChain, LangGraph i provider paketa;
- da li FastEmbed sparse model radi lokalno u našem okruženju;
- da li `langchain-qdrant` izlaže native hybrid kontrole koje su nam potrebne;
- da li instalacija uvodi konflikt sa postojećim Transformers i NumPy
  verzijama.

Framework zavisnosti je bolje staviti u optional grupu, na primer
`.[assistant]`, dok corpus build i retrieval baseline ostaju instalabilni bez
njih dok migracija ne bude prihvaćena.

## 18. Faze implementacije

### Faza 0: usaglašavanje odluka

Pre koda potvrditi:

- Qdrant Local embedded/client path naspram lokalnog Docker servera;
- prvi sparse encoder;
- da li Qdrant baza pripada u `data/processed/qdrant/`;
- kriterijume za prihvatanje eventualne male retrieval regresije;
- da li stari BM25S ostaje kao fallback posle prihvatanja Qdranta.

Preporuka plana je Qdrant Local bez Docker zahteva za prvu iteraciju i
Qdrant/FastEmbed BM25 kao prvi sparse kandidat.

### Faza 1: dependency i API spike

Minimalan, izolovan probe treba da potvrdi:

- kreiranje persistent local kolekcije;
- upload postojećeg 1.024-dimenzionalnog vektora;
- sparse point i sparse query;
- native RRF;
- metadata filter;
- restart i ponovno otvaranje baze;
- Python 3.13 kompatibilnost.

Spike nije aktivna arhitektura i ne menja postojeće search/evaluate skripte.

### Faza 2: collection schema, mapper i manifest

Implementirati point mapping, deterministic ID, collection lifecycle i
manifest validaciju. Završiti integritet i restart testove.

### Faza 3: dense import i parity

Importovati postojeće embeddinge i pokrenuti E1 evaluaciju. Ne nastavljati sa
LLM orkestracijom dok dense mapiranje nije pouzdano.

### Faza 4: sparse i native RRF

Implementirati E2 i E3. Uporediti agregatne i per-query rezultate. Zadržati
BM25S baseline dok Qdrant varijanta ne prođe Gate 3.

### Faza 5: retrieval servis i CLI backend

Uvesti stabilni BankScope retrieval servis. `search.py` i `evaluate.py` treba
da mogu eksplicitno da izaberu baseline ili Qdrant backend tokom tranzicije.
Default se menja tek nakon zabeležene odluke.

### Faza 6: LangChain tool sloj

Definisati jedan retrieval tool sa validiranim inputom i strukturiranim
evidence outputom. LangChain ne menja indexing ili evidence identitete.

### Faza 7: minimalni LangGraph

Povezati model node, ToolNode, evidence validation i final answer node. Dodati
iteration limit, error putanju i unsupported/insufficient-evidence status.

### Faza 8: end-to-end answer evaluacija

Tek kada retrieval ostane stabilan, dodati gold/acceptable answers i meriti:

- numeričku answer accuracy;
- tačnost godine i jedinice;
- citation correctness;
- citation completeness;
- groundedness;
- refusal accuracy;
- cross-bank completeness.

## 19. Rizici i mitigacije

### Sparse promena pogorša BM25 rezultat

Mitigacija: sačuvati E0 baseline, pokrenuti per-query diff i ne uklanjati
BM25S pre prihvatanja E3.

### Qdrant ubrza search, ali query encoder ostane spor

Mitigacija: meriti komponente odvojeno, model učitati jednom po procesu i ne
pripisivati encoder latenciju bazi.

### Local Qdrant i server Qdrant se ponašaju različito

Mitigacija: local je cilj trenutne faze; server kompatibilnost se proverava
posebnim testom samo ako deployment postane zahtev.

### LangChain sakrije native Qdrant opcije

Mitigacija: Qdrant client ostaje u infrastructure sloju; LangChain tool poziva
BankScope servis umesto da diktira core query.

### LangGraph uvede nepotrebnu složenost

Mitigacija: početni graph ima jedan model node, jedan tool node i eksplicitna
stop pravila. Novi node se dodaje samo zbog merljivog ponašanja.

### Framework verzije brzo menjaju API

Mitigacija: kompatibilne verzije se pinuju nakon probe-a, framework objekti se
ne šire kroz domain sloj, a testovi pokrivaju granice adaptera.

### Baza postane drugi izvor istine

Mitigacija: `chunks.jsonl`, `tables.jsonl`, `manifest.json` i
`embeddings.npz` ostaju kanonski artefakti. Qdrant se može potpuno obnoviti iz
njih i odbija se kada hash ugovor ne odgovara.

## 20. Dokumentacija i odluke

Nakon prihvatanja Qdrant retrievala treba:

1. dodati decision record sa poređenjem E0-E3;
2. ažurirati `docs/data_pipeline.md` stvarnim aktivnim tokom;
3. ažurirati `docs/roadmap.md` tek kada se status faze promeni;
4. ažurirati README komande i setup;
5. zabeležiti generated Qdrant folder u `.gitignore` ako već nije pokriven;
6. dokumentovati kako se baza obnavlja, proverava i briše bez rizika po source
   corpus.

Plan sam po sebi ne menja postojeću arhitektonsku odluku. Aktivni default se
menja tek kada evaluacija i decision record to opravdaju.

## 21. Predloženi neposredni sledeći korak

Kada plan bude odobren, prvi implementacioni korak treba da bude samo Faza 1:
kratak Qdrant Local compatibility spike u izolovanom test/probe kontekstu.

Taj korak treba da odgovori na pet pitanja pre šire migracije:

1. Da li izabrane verzije rade na Pythonu 3.13 bez dependency konflikta?
2. Da li persistent Qdrant Local pouzdano čuva i ponovo otvara kolekciju?
3. Da li možemo importovati postojeći Qwen vektor bez re-embedovanja?
4. Da li odabrani sparse encoder i Qdrant-native RRF rade kroz potreban API?
5. Da li LangChain adapter izlaže dovoljno kontrole ili core mora koristiti
   direktan Qdrant client?

Tek nakon odgovora na ova pitanja treba menjati aktivni retrieval kod.

## 22. Referentna dokumentacija

- Qdrant hybrid queries i fusion:
  <https://qdrant.tech/documentation/search/hybrid-queries/>
- Qdrant dense/sparse search:
  <https://qdrant.tech/documentation/search/>
- Qdrant fundamentals i granice full-text podrške:
  <https://qdrant.tech/documentation/faq/qdrant-fundamentals/>
- Qdrant i LangChain integracija, uključujući local i hybrid režime:
  <https://qdrant.tech/documentation/frameworks/langchain/>
- LangChain tools:
  <https://docs.langchain.com/oss/python/langchain/tools>
- LangChain modeli i provider interfejsi:
  <https://docs.langchain.com/oss/python/langchain/models>
- LangGraph overview:
  <https://docs.langchain.com/oss/python/langgraph/overview>
- LangGraph graph API:
  <https://docs.langchain.com/oss/python/langgraph/graph-api>
- LangGraph persistence:
  <https://docs.langchain.com/oss/python/langgraph/persistence>
- Pydantic modeli i JSON Schema:
  <https://pydantic.dev/docs/validation/latest/concepts/models/>
