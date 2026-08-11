# BankScope: vodič kroz projekat i kod

## 1. Šta je BankScope

BankScope je studentski RAG projekat za pretragu najnovijih lokalno preuzetih
10-K izveštaja deset američkih banaka. Trenutno je završen deo sistema koji:

1. preuzima izveštaje sa SEC EDGAR-a;
2. pretvara HTML u strukturisan tekst i cele tabele;
3. pravi embeddinge za semantičku pretragu;
4. indeksira embeddinge u lokalnom Qdrant-u;
5. pretražuje korpus dense, BM25 ili hybrid metodom;
6. meri kvalitet retrieval-a na unapred označenim pitanjima;
7. generiše proverljiv odgovor iz hidriranih dokaza, uz citate ili abstention.

Retrieval i prvi single-bank generation tok sada rade. GPT-5.1 v2 frozen baseline
je završen bez schema/format grešaka i prošao je sve answer-quality provere, ali
gate ostaje neuspešan zbog jednog zaokruženog dodatnog citata. GPT-5.1 zato nije
default; taj citation problem je odložen, dok se nastavlja razvoj bank resolvera,
conversation history-ja i UI-ja.

## 2. Najvažnija mentalna slika

```text
config/banks.yaml
        |
        v
scripts/download.py
        |
        +--> data/filings.json
        +--> data/raw/sec/**/*.htm
                    |
                    v
          scripts/build_corpus.py
                    |
                    +--> data/processed/chunks.jsonl
                    +--> data/processed/tables.jsonl
                    +--> data/processed/manifest.json
                                 |
                                 v
                         scripts/embed.py
                                 |
                                 +--> embeddings.npz
                                 |
                                 v
                      scripts/build_qdrant.py
                                 |
                                 +--> qdrant/
                                 +--> qdrant_manifest.json
                                 |
              +------------------+------------------+
              v                                     v
      scripts/search.py                    scripts/evaluate.py
```

`scripts/` sadrži komande koje korisnik pokrece. `src/bankscope/` sadrži
ponovo upotrebljivu poslovnu logiku koju te komande pozivaju. `data/` sadrži
ulaze, medurezultate i rezultate.

## 3. Kako radi RAG deo koji je sada implementiran

RAG znaci Retrieval-Augmented Generation:

- **Retrieval** pronalazi relevantne delove izvora za pitanje.
- **Augmentation** prosleduje pronadene dokaze modelu zajedno sa pitanjem.
- **Generation** sastavlja odgovor iz tih dokaza.

BankScope je trenutno završio retrieval sloj. Za pitanje pravi embedding upita,
traži semanticki slicne zapise u Qdrant-u, paralelno radi leksicku BM25S
pretragu, zatim spaja rang-liste pomocu RRF-a. Ako pogodak predstavlja tabelu,
opis tabele se zamenjuje kompletnom originalnom Markdown tabelom pre prikaza.

### Zašto postoje i opis i cela tabela

Velike finansijske tabele nisu dobar embedding input: pune su brojeva, cesto su
dugacke i lako prelaze token limit. Zato postoje dva sloja:

- `chunks.jsonl` sadrži kratak opis tabele namenjen pretrazi;
- `tables.jsonl` sadrži celu tabelu namenjenu citiranju i odgovaranju.

Veza je stabilni `table_id`, koji je ujedno `target_chunk_id` retrieval zapisa.
Pretražuje se opis, ali se korisniku vraca originalna tabela.

## 4. Folderi u korenu projekta

### `src/`

Glavni Python paket. Ovo je kod koji predstavlja samu aplikaciju, a ne jednu
konkretnu CLI komandu. `pyproject.toml` koristi `src` layout, pa se paket
`bankscope` instalira iz `src/bankscope`.

Prednosti `src` layout-a su da testovi koriste instalirani paket na isti nacin
kao stvarna aplikacija i da se slucajno ne importuje kod direktno iz root-a.

### `scripts/`

Izvršne komande pipeline-a: download, izgradnja korpusa, embedding, Qdrant,
pretraga i evaluacija. One orkestriraju funkcije iz `src/` i rade sa fajlovima.

### `data/`

Podaci kroz sve faze pipeline-a. Mali ugovori (`filings.json` i evaluaciona
pitanja) prate se u Git-u; veliki preuzeti i generisani artefakti su lokalni i
ignorisani.

### `config/`

Staticka konfiguracija banaka. Trenutno sadrži `banks.yaml`.

### `tests/`

Unit i integracioni testovi za aktivni kod. Testira validaciju, deterministicnost,
bezbedno pisanje fajlova, table hydration, sve retrievere i metrike.

### `docs/`

Projektna dokumentacija, odluke i roadmap. `decisions/` su ADR zapisi: objašnjavaju
zašto je izabran odredeni parser ili backend, zajedno sa izmerenim rezultatima.

### `notebooks/`

Aktivni Colab notebook za GPU embedding i evaluaciju. Nije runtime aplikacije;
služi za reprodukovanje compute-intenzivnog eksperimenta na T4 GPU-u.

### `sandbox/`

Arhiva starog i eksperimentalnog koda. Ništa odavde se ne importuje u aktivni
projekat.

- `legacy_builtin/`: stari BeautifulSoup parser i table-proxy pipeline;
- `legacy_v3/`: stari sec2md row/cell locator pipeline i rezultati;
- `experiments/`: Docling, XBRL, notebook i Supabase probe;
- `docs/`: stara dokumentacija.

Ovaj kod cuva istoriju odluka, ali može imati zastarele putanje i zavisnosti.

### `artifacts/`

Lokalni test i transportni artefakti, na primer pytest Qdrant direktorijumi i
Colab bundle. Nije deo aktivnog izvornog koda.

### `experiments/`

Trenutno nema aktivne fajlove. Eksperimenti koji su sacuvani nalaze se pod
`sandbox/experiments/`.

### `.venv/`, `.ruff_cache/`, `.tmp_pytest_*`, `pytest-cache-files-*`

Lokalno razvojno okruženje i keševi alata. Mogu se ponovo napraviti i ne nose
poslovnu logiku. `.git/` je interna Git baza repozitorijuma.

## 5. `src/bankscope/` detaljno

Prazni `__init__.py` fajlovi samo oznacavaju Python pakete. Logika je u sledecim
modulima.

### `config/settings.py`

Centralizuje konfiguraciju okruženja pomocu Pydantic Settings-a.

- `PROJECT_ROOT` racuna root projekta iz lokacije modula.
- `ApplicationSettings` ucitava `.env` i definiše SEC, registry, raw-data i
  opcione OpenAI parametre.
- `sec_user_agent` mora sadržati email, jer SEC zahteva identifikovan User-Agent.
- brzina SEC poziva mora biti veca od nule i najviše 10 zahteva u sekundi;
- `SecretStr` sprecava slucajno prikazivanje OpenAI kljuca;
- `get_settings()` je keširan, pa se konfiguracija cita samo jednom po procesu.

### `sec/company_registry.py`

Validira `config/banks.yaml`.

- `BankCompany` cuva ticker, desetocifreni CIK, pravno ime i enabled zastavicu;
- ticker se normalizuje na velika slova i dozvoljava slova, cifre, tacku i crticu;
- modeli su `frozen`, pa se posle validacije ne menjaju;
- `BankRegistry` odbija duple tickere, duple CIK vrednosti i registry bez aktivne
  banke;
- `load_bank_registry()` ucitava YAML i vraca validiran Pydantic model.

### `io.py`

Zajednicke, strogo validirane I/O funkcije.

- `read_jsonl()` cita JSON Lines, preskace prazne redove i prijavljuje tacan broj
  neispravnog reda;
- `write_jsonl()` piše prvo privremeni fajl, pa ga atomski zamenjuje pomocu
  `os.replace`; polovicno napisan izlaz zato ne zamenjuje dobar fajl;
- `sha256_file()` racuna SHA-256 u blokovima;
- `load_embedding_archive()` ucitava NPZ i proverava obavezna polja, 2D oblik,
  `float32`, konacne vrednosti, jedinicnu normu, jedinstvene ID-jeve, njihov
  redosled i validan hash izvornog `chunks.jsonl`.

Ovaj modul sprovodi važan princip projekta: zastareo ili pogrešno uparen artefakt
mora izazvati grešku umesto tihog pogrešnog rezultata.

### `parsing/tables.py`

Obraduje tabele koje emituje `sec2md==0.1.23`.

- `make_chunk_id()` pravi stabilan SHA-256 identitet iz accession broja, varijante,
  sadržaja, element ID-jeva i child kljuca;
- `parse_markdown_table_matrices()` cita Markdown tabele bez menjanja izvora;
- `extract_sec2md_table_grids()` iz anotiranog HTML-a uzima celijske mreže i
  pravilno razrešava `rowspan`/`colspan` preko sec2md `TableParser`-a;
- `classify_table()` klasifikuje tabelu kao `layout`, `index`, `glossary`,
  `narrative_table` ili `data_table`;
- layout i index tabele se cuvaju radi audit-a, ali nisu retrieval eligible;
- pomocne funkcije izvlace naslov, jedinicu, periode, kolone i znacajne row labele;
- `build_local_description()` pravi deterministicki opis bez kopiranja numerickog
  tela tabele;
- `build_openai_description()` opciono poziva Responses API za kratak semanticki
  synopsis; greška se ne skriva lokalnim fallback-om;
- `describe_table()` u OpenAI modu spaja deterministicki indeks i LLM synopsis i
  beleži provenance;
- `build_table_record()` cuva kompletnu tabelu, celijske matrice, tip, ID i SEC
  metapodatke.

### `parsing/corpus.py`

Spaja sec2md elemente u konacni retrieval korpus.

- konstante postavljaju cilj od 512 tokena, maksimum 1.024, overlap 64 i krajnji
  embedding maksimum 2.048 tokena;
- uklanja navigaciju, sadržaj i page furniture koji nisu korisni dokazi;
- cuva redosled elemenata i deduplikuje elemente koji se pojavljuju na više strana;
- prepoznaje headinge, SEC `Item` oznake i glossary unose;
- `split_to_token_limit()` deli samo narativ, uz mali overlap radi konteksta;
- `_build_embedding_text()` dodaje banku, entitet, godinu, tip dokaza, strane i
  sekciju ispred sadržaja;
- `_build_text_record()` pravi narativni retrieval zapis;
- `_build_table_chunk()` pravi retrieval opis koji pokazuje na celu tabelu;
- `validate_corpus()` proverava jedinstvenost, validne tipove, postojanje table
  targeta, tacno jedan opis po eligible tabeli i token limit;
- `build_corpus()` je glavni algoritam: prolazi kroz elemente u izvornom redosledu,
  prazni narrative buffer na headingu/limitu/tabeli, posebno deli glossary, cuva
  svaku celu tabelu i indeksira samo eligible tabele.

### `retrieval/hybrid_retriever.py`

Referentna lokalna implementacija dense + BM25S + RRF retrieval-a.

- `normalize_lexical_text()` normalizuje Unicode, crtice i uklanja zareze izmedu
  cifara, pa su `1,000` i `1000` leksicki kompatibilniji;
- finansijski tokenizer cuva tokene kao `10-k`, `cet1`, procente i slicne oblike;
- `HybridRetriever` validira zapise, po potrebi normalizuje embedding matricu i
  gradi BM25S Lucene indeks;
- `search_dense()` racuna cosine slicnost matricnim množenjem;
- `search_bm25()` vraca leksicki rang;
- oba puta podržavaju `ticker` i `record_type` filter;
- `_make_result()` hidrira table hit iz `tables.jsonl`;
- `reciprocal_rank_fusion()` sabira `1 / (rrf_k + rank)` iz dense i BM25 rangova,
  deduplikuje po dokaznom ID-u i deterministicki razrešava izjednacenja;
- `search_hybrid()` prvo uzima širi candidate window, pa tek onda vraca top N.

### `retrieval/qdrant_retriever.py`

Cita persistentni lokalni Qdrant indeks.

- manifest se validira pre otvaranja baze;
- proveravaju se broj tacaka, dense dimenzija i prisustvo sparse vektora;
- opciono se proverava hash `tables.jsonl`;
- `search_dense()` koristi named vector `dense`;
- `search_bm25()` koristi FastEmbed dokument sa modelom `Qdrant/bm25` i named
  sparse vector `sparse`;
- `search_hybrid()` koristi dva Qdrant prefetch-a i nativni Qdrant RRF;
- rezultati imaju isti oblik kao baseline i cele tabele se ponovo hidriraju;
- `close()` eksplicitno zatvara lokalnu bazu, što je narocito važno na Windows-u.

### `retrieval/mixed_retriever.py`

Aktivna arhitektura retrieval-a.

- dense poziv delegira Qdrant-u;
- BM25 poziv delegira lokalnom `HybridRetriever`-u;
- hybrid uzima po 30 kandidata iz oba izvora i spaja ih aplikacionim RRF-om;
- validira query i RRF parametre.

Ovo je kompromis izmedu zahteva da embedding bude u VectorDB-u i boljeg kvaliteta
lokalnog BM25S/RRF puta.

### `evaluation/retrieval_metrics.py`

Racuna retrieval metrike.

- deduplikuje ponovljene target ID-jeve;
- `Hit@k` kaže da li je bar jedan relevantan dokaz u prvih k;
- `Recall@k` kaže koji deo svih relevantnih dokaza je pronaden;
- `MRR@10` nagraduje raniju poziciju prvog relevantnog dokaza;
- evidence-group metrike proveravaju da li cross-bank pitanje pokriva svaku
  potrebnu banku/grupu, a ne samo bilo koji relevantan dokaz.

## 6. Sve aktivne skripte

### `scripts/download.py`

Za svaku enabled banku poziva SEC submissions API, bira najnoviji obicni `10-K`
(ne `10-K/A`), bezbedno proverava ime dokumenta, preuzima primarni HTML i atomski
piše `data/filings.json`. Poštuje User-Agent, timeout i rate limit. Kada se koristi
`--ticker`, ažurira samo tu banku bez brisanja ostalih manifest redova.

### `scripts/build_corpus.py`

Ucitava filing manifest, otvara svaki HTML, pokrece sec2md parser i prosleduje
strane u `build_corpus()`. Ucitava Qwen tokenizer samo radi preciznog brojanja
tokena. Piše `chunks.jsonl`, `tables.jsonl` i manifest sa hash-evima, verzijama i
brojevima zapisa. Odbija overwrite bez `--overwrite`. Filtrirani smoke build mora
imati poseban `--output-dir`, da ne pregazi puni korpus.

### `scripts/embed.py`

Ucitava `embedding_text` u istom redosledu kao JSONL, lazy ucitava
`Qwen/Qwen3-Embedding-0.6B`, koristi GPU ako postoji, proverava dužinu bez tihog
truncation-a i pravi normalizovane `float32` embeddinge. NPZ sadrži matricu,
redosled ID-jeva, model, revision i hash ulaza. `--limit` je smoke režim i ništa
ne upisuje.

### `scripts/build_qdrant.py`

Validira medusobnu uskladenost korpusa, tabela, embeddinga i manifesta. Kreira
lokalnu kolekciju `bankscope_retrieval` sa:

- dense cosine vektorom dimenzije 1.024;
- sparse BM25 vektorom sa IDF modifier-om;
- payload poljima potrebnim za filtere i prikaz;
- deterministickim UUID point ID-jevima izvedenim iz `record_id`.

Po završetku proverava broj tacaka i šemu, pa piše `qdrant_manifest.json` sa
hash-evima izvora. Postojecu kolekciju menja samo uz eksplicitni `--recreate`.

### `scripts/search.py`

Glavni rucni search CLI. Bira backend (`mixed`, `baseline`, `qdrant`) i metod
(`hybrid`, `dense`, `bm25`). Dense/hybrid režim ucitava tacnu revision verziju
embedding modela i enkoduje pitanje. BM25 ne ucitava Torch ni embedding arhivu.
Podržava filtere po banci i tipu zapisa, zatvara Qdrant i štampa čist JSON sa
rangom, skorovima, metapodacima i hidriranim dokazom.

### `scripts/evaluate.py`

Ucitava frozen qrels iz `queries.jsonl`, validira svaki query i target, enkoduje
samo answerable pitanja i izvršava izabrane metode/backende. Za svako pitanje
cuva top 10 i metrike, zatim racuna zbirni Hit/Recall/MRR, group coverage i
latenciju. `--backend all` poredi sva tri backenda, proverava Qdrant quality gate
i proverava da li mixed daje iste rang-liste kao baseline. Ambiguous i unsupported
pitanja se beleže kao diagnostika, ali ne ulaze u retrieval metrike.

### `scripts/answer.py`

Izvršava aktivni mixed hybrid retrieval za jednu obavezno izabranu banku, zatim
prosleđuje samo hidrirane dokaze Chat Completions-kompatibilnom modelu. Pre API poziva
proverava entitet, tip dokaza i traženi period. Izlaz je JSON sa statusom
`supported`, `ambiguous` ili `unsupported`; citati se prihvataju samo ako pokazuju
na prosleđeni dokaz, a filing, stranica i URL se grade iz lokalnih metapodataka.

### `scripts/smoke_qdrant.py`

Bira nasumicno answerable pitanje, izvršava jedan Qdrant BM25/dense/hybrid upit i
meri odvojeno vreme enkodovanja, otvaranja baze i samog query-ja. Namenjen je
brzoj lokalnoj proveri, ne zvanicnoj evaluaciji.

### `scripts/benchmark_query_embeddings.py`

Meri cold model load, prvi query, pojedinacne query-je i batch throughput na CPU
ili CUDA uredaju. Ne menja projektne podatke.

## 7. `data/` detaljno

### `data/filings.json`

Manifest deset 2025 10-K filing-a. Svaki red ima ticker, CIK, pravno ime, formu,
accession, filing/report datum, primarni dokument, SEC URL i lokalnu putanju.

### `data/raw/sec/`

Originalni HTML fajlovi, organizovani kao `CIK/accession/dokument`. To je izvor
istine za ponovno parsiranje. Trenutno postoji po jedan filing za svih deset
banaka. WFC i USB primarni dokumenti su parcijalni jer upucuju na odvojeni annual
report attachment.

### `data/processed/chunks.jsonl`

Jedan retrieval zapis po redu. Kljucna polja su:

- `record_id`: identitet i red embeddinga;
- `target_chunk_id`: identitet dokaza koji evaluacija koristi;
- `record_type`: `text` ili `table`;
- `embedding_text`: tekst koji se indeksira;
- `document`: narativni dokaz ili opis tabele;
- `metadata`: banka, filing, parser, strane, sekcija, XBRL tagovi i provenance.

Trenutno: 5.565 redova = 4.009 tekstualnih + 1.556 table-description zapisa.

### `data/processed/tables.jsonl`

Canonical table store. Cuva 1.783 cele Markdown tabele, cell matrices,
klasifikaciju, eligibility i izvorne metapodatke. Njih 1.556 je retrieval eligible;
227 layout/index tabela se samo cuva radi audit-a.

### `data/processed/lexical_glossary_locators_v1.jsonl`

Verzionisani lexical-only zapisi za acronym/glossary parove. Svaki locator pokazuje na postojeci
whole-table `target_chunk_id`; BM25 deduplikuje parent tabelu pre limita, a korisniku se uvek
vraca puna tabela. Dense embeddings i Qdrant kolekcija se zbog ovih locatora ne menjaju.

### `data/processed/manifest.json`

Dokaz kako je korpus napravljen: verzije parsera i tokenizer-a, token limit,
opisni režim, broj filing-a/zapisa/tabela, per-bank statistika i SHA-256 izlaza.
Najduži trenutni embedding input ima 1.049 od dozvoljenih 2.048 tokena.

### `data/processed/embeddings.npz`

NumPy arhiva: `embeddings` oblika `(5565, 1024)`, `record_ids`, ime modela,
revision i hash `chunks.jsonl`. Red N u matrici pripada redu N u chunks fajlu.

### `data/processed/qdrant/`

Persistentna lokalna Qdrant baza sa 5.565 tacaka. Nije rucno editovan format;
ponovo se gradi preko `build_qdrant.py`.

### `data/processed/qdrant_manifest.json`

Ugovor Qdrant baze: kolekcija, broj tacaka, dense model/revision/dimenzija,
sparse model i hash-evi chunks/tables/embeddings izvora.

### `data/evaluation/queries.jsonl`

Frozen skup od 30 pitanja. Sadrži status, pitanje, filtere, tip pitanja, relevantne
target ID-jeve, gold answer i beleške. Od toga je 28 answerable, jedno ambiguous i
jedno unsupported. Neka cross-bank pitanja imaju `required_evidence_groups`.

### `data/evaluation/results/`

`retrieval.json` je kompletan zbirni i per-query rezultat. `run_provenance.json`
beleži Colab/T4 okruženje, parametre i hash-eve korišćenih artefakata.
`retrieval-glossary-locators-v1.json` je odvojeni lokalni v2 retrieval rezultat; istorijski
`retrieval.json` ostaje netaknut.

## 8. Konfiguracioni i root fajlovi

### `config/banks.yaml`

Registry deset banaka: JPM, BAC, C, WFC, USB, PNC, TFC, GS, ALLY i LOB. Ticker i
CIK su tehnicki kljucevi; `enabled` kontroliše koje banke ulaze u download.

### `.env.example` i `.env`

Primer navodi SEC User-Agent, rate/timeout, registry/raw putanje i opcioni OpenAI
kljuc/model. Pravi `.env` je lokalna tajna i ne treba ga commit-ovati niti deliti.

### `pyproject.toml`

Definiše Python 3.13 projekat, runtime i optional zavisnosti, `src` package
discovery, pytest putanje i Ruff pravila. Glavne biblioteke su Pydantic, sec2md,
Transformers/SentenceTransformers, NumPy, BM25S i Qdrant client.

### `.gitignore`

Sprecava commit tajni, virtualnog okruženja, keševa i velikih generisanih podataka.

### `README.md`, `LICENSE`

README je kratko operativno uputstvo i trenutni dizajn. LICENSE definiše pravne
uslove korišćenja repozitorijuma.

## 9. Testovi

- `test_settings.py`: env validacija, SEC parametri, skrivanje tajne i caching;
- `test_company_registry.py`: normalizacija i duplikati registry-ja;
- `test_download.py`: izbor najnovijeg 10-K, manifest merge i path traversal;
- `test_io.py`: JSONL, atomsko pisanje, SHA-256 i NPZ ugovori;
- `test_tables.py`: cuvanje svih tabela, eligibility i OpenAI provenance/failure;
- `test_corpus.py`: stabilni ID-jevi, lossless table store, determinizam i limiti;
- `test_embed.py`: zabrana tihog truncation-a;
- `test_hybrid_retriever.py`: dense/BM25/RRF, filteri, hydration i invalidni ulazi;
- `test_qdrant_retriever.py`: persistentni dense/sparse/hybrid query i hydration;
- `test_mixed_retriever.py`: delegiranje grana i application RRF;
- `test_retrieval_metrics.py`: qrel validacija, metrike i BM25 bez Torch-a;
- `test_benchmark_query_embeddings.py`: ulazi benchmark skripte.
- `test_answer_generator.py`, `test_answer_pipeline.py`: grounded odgovor i reusable pipeline;
- `test_answer_metrics.py`, `test_answer_evaluator.py`, `test_semantic_judge.py`:
  generation scope, metrike i validacija savetodavnog judge-a.

Aktivni Python kod u `src/`, `scripts/` i `tests/` trenutno prolazi Ruff lint i
format proveru. U ovoj ogranicenoj sesiji 70 testova je prošlo, dok 15 testova koji koriste pytest
`tmp_path` nisu mogli da se podese zbog zabrane pristupa temp direktorijumu; to je
ogranicenje okruženja, ne zabeležen assertion failure. Globalni Ruff nad celim
repozitorijumom trenutno prijavljuje samo tri lint/format problema u Colab notebooku.

## 10. Gde smo tacno stigli

Završeno:

- registry i SEC acquisition za deset banaka;
- izbor i pinovanje sec2md parsera;
- jednostavan aktivni pipeline i izolovan legacy kod;
- whole-table model sa odvojenim opisom za retrieval;
- Qwen embedding pipeline i GPU artefakt;
- baseline, puni Qdrant i mixed retrieval;
- frozen evaluacija i dokumentovana backend odluka;
- hash/order/schema zaštite i testovi.
- single-bank answer generation iz hidriranih dokaza, sa abstention-om i citatima.
- generation evaluator sa odvojenim determinističkim i savetodavnim metrikama.

Aktivni default je:

```text
Qdrant dense + BM25S lexical + application RRF = mixed hybrid
```

Na 28 answerable pitanja mixed hybrid ima:

- Hit@1: 10/28;
- Hit@5: 25/28;
- Hit@10: 26/28;
- MRR@10: 0,596;
- mean Recall@10: 0,855;
- prosecnu lokalnu retrieval latenciju oko 84,5 ms u zabeleženom run-u.

Mixed rang-liste su identicne baseline rang-listama u dense, BM25 i hybrid
režimu. Puni Qdrant hybrid zadržava Hit@10 = 26/28, ali MRR@10 pada na 0,547 i
zato nije prošao definisani MRR prag.

Sledeće faze su:

1. deterministički prepoznati banku iz naziva, aliasa ili tickera pre retrieval-a;
2. conversation history koji nasleđuje banku iz sesije bez kontaminacije retrieval query-ja;
3. jednostavan lokalni chat UI bez ručnog izbora banke;
4. kasnije rešiti suvišne, nedovoljno precizne citate i eventualno ponoviti frozen run
   samo uz posebno odobrenje;
5. odvojeno izveštavanje retrieval i generation kvaliteta.

## 11. Kako pokrenuti projekat

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Zatim, iz root-a:

```powershell
python scripts/download.py
python scripts/build_corpus.py --overwrite
python scripts/embed.py --overwrite
python scripts/build_qdrant.py
python scripts/search.py "How does JPMorgan Chase define cybersecurity risk?" --ticker JPM
python scripts/answer.py "How does JPMorgan Chase define cybersecurity risk?"
python scripts/evaluate.py
python scripts/evaluate_answers.py
```

Korisne krace provere:

```powershell
python scripts/download.py --ticker JPM
python scripts/build_corpus.py --ticker JPM --output-dir data/processed/smoke-jpm --overwrite
python scripts/embed.py --limit 10
python scripts/smoke_qdrant.py --mode bm25 --seed 1
python -m pytest
python -m ruff check src scripts tests
```

## 12. Najvažniji pojmovi za ucenje

- **CIK**: stabilni SEC identifikator kompanije.
- **Accession number**: identitet konkretnog SEC podneska.
- **Chunk**: ogranicen deo teksta koji se zasebno indeksira.
- **Embedding**: numericki vektor semantickog znacenja teksta.
- **Dense retrieval**: rangiranje po slicnosti embedding vektora.
- **BM25**: leksicko rangiranje zasnovano na terminima i njihovoj retkosti.
- **Hybrid retrieval**: kombinovanje dense i leksickog rezultata.
- **RRF**: spajanje rang-lista bez direktnog poredenja njihovih razlicitih skorova.
- **Hydration**: zamena table opisa kompletnom izvornom tabelom.
- **Qrel**: rucno oznacena veza izmedu pitanja i relevantnog dokaza.
- **Provenance**: trag od rezultata do izvora, modela, verzije i hash-a.
- **Invariant**: uslov koji uvek mora važiti, na primer isti red chunks i embeddinga.
- **Smoke test**: mala brza provera da glavni put radi.
- **Quality gate**: unapred definisan prag koji promena mora proci da bi bila prihvacena.

## 13. Ogranicenja i važne napomene

- USB i WFC korpusi su parcijalni zbog odvojenih annual-report attachmenta.
- Jedna tabela koju sec2md emituje kao više continuation elemenata ostaje više
  tabela; BankScope samo garantuje da jedan emitovani element nece dalje deliti.
- Retrieval score nije verovatnoca da je pitanje odgovorivo.
- Ambiguous i buduće/unsupported pitanje i dalje mogu dobiti uverljivo izgledajuće
  retrieval rezultate; generation sloj zato primenjuje support check i abstention.
- Frozen skup ima samo 30 pitanja i ne treba ga koristiti za nekontrolisano tuning
  ponavljanje, jer bi se sistem overfit-ovao na evaluaciju.
- Stari istorijski commit je sadržao Hugging Face token; dokumentovana odluka kaže
  da taj token treba opozvati/rotirati cak iako ga nema u trenutnom stablu.

## 14. Najkraci rezime

BankScope je sada robustan retrieval sistem nad 10-K izveštajima deset banaka sa
prvim bezbednim single-bank generation tokom i implementiranim evaluatorom.
`scripts/` pokreće pipeline, `src/` nosi validiranu logiku, `data/` čuva izvore i
artefakte, `tests/` štiti ugovore, a `sandbox/` čuva istoriju van runtime-a.
Prvi generation baseline je zabeležen nad 26 in-scope pitanja: 24 su evaluirana,
a dve model-format/citation greške su sačuvane kao eksplicitni rezultati. GPT-5.1
v2 kandidat je zatim završio svih 26 pitanja i prošao sve answer-quality provere,
ali je post-run citation audit ostao na 24/25 zbog jednog zaokruženog dodatnog
dokaza. Default model ostaje nepromenjen, citation problem je odložen, a razvoj
bank resolvera, conversation history-ja i UI-ja može da se nastavi.
