# Qdrant plan za BankScope

## 1. Namena dokumenta

Ovaj dokument je detaljan plan za uvođenje Qdranta u BankScope. Fokus je samo
na vector database i retrieval sloju:

- izbor načina lokalnog pokretanja;
- model kolekcije, vektora, pointova i payloada;
- import postojećih Qwen dense embeddinga;
- generisanje i čuvanje Qdrant sparse vektora;
- dense, sparse i native RRF upiti;
- metadata filteri;
- table evidence hydration;
- verzionisanje, integritet, obnavljanje i backup;
- testiranje kvaliteta i brzine;
- mogućnosti koje Qdrant daje za kasnije unapređenje.

LangChain, LangGraph, OpenAI answer generation i UI nisu predmet
implementacije u ovom dokumentu. Oni se pominju samo tamo gde Qdrant mora da
obezbedi stabilan interfejs za buduće potrošače.

Plan ne pretpostavlja da će Qdrant automatski poboljšati kvalitet ili latenciju.
Nova implementacija postaje aktivna tek nakon provere integriteta, retrieval
evaluacije i merenja latencije.

## 2. Trenutno stanje koje Qdrant mora da očuva

Potvrđeni aktivni artefakti su:

```text
data/processed/chunks.jsonl
data/processed/tables.jsonl
data/processed/manifest.json
data/processed/embeddings.npz
data/evaluation/queries.jsonl
data/evaluation/results/retrieval.json
data/evaluation/results/run_provenance.json
```

Potvrđene dimenzije i ugovori:

- 5.565 retrieval zapisa;
- 4.009 narrative text zapisa;
- 1.556 table retrieval opisa;
- 1.783 cele tabele u table store-u;
- 1.024 dense dimenzije;
- `float32`, normalizovani Qwen vektori;
- model `Qwen/Qwen3-Embedding-0.6B`;
- model revision
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`;
- svaki dense red odgovara `record_id` vrednosti u istom redosledu;
- table retrieval point pretražuje opis i metadata, a dokaz je cela tabela iz
  `tables.jsonl`;
- postojeći lexical retrieval koristi BM25S, `lucene` scoring i namensku
  tokenizaciju;
- postojeći hybrid spaja dense i BM25S rangove preko RRF-a.

Prihvaćeni GPU retrieval baseline:

| Metoda | Hit@1 | Hit@5 | Hit@10 | MRR@10 | Mean Recall@10 |
|---|---:|---:|---:|---:|---:|
| Dense | 10/28 | 22/28 | 23/28 | 0,522 | 0,669 |
| BM25S | 11/28 | 25/28 | 26/28 | 0,562 | 0,787 |
| Hybrid RRF | 11/28 | 25/28 | 26/28 | 0,614 | 0,855 |

Qdrant baza je izvedeni indeks. Kanonski izvori ostaju JSONL, manifest i NPZ
artefakti. Baza mora moći potpuno da se obriše i ponovo izgradi iz tih izvora.

## 3. Cilj Qdrant implementacije

Qdrant sloj treba da omogući:

1. persistent lokalnu bazu;
2. jedan point po retrieval zapisu;
3. postojeći Qwen dense vektor bez re-embedovanja;
4. jedan Qdrant/BM25 sparse vektor po istom zapisu;
5. cosine dense search;
6. BM25 sparse search;
7. native Qdrant RRF nad dense i sparse prefetch rangovima;
8. ticker, record type i druge proverene metadata filtere;
9. vraćanje stabilnih BankScope evidence identiteta;
10. table hydration iz kanonskog table store-a;
11. odbijanje zastarele ili nekompatibilne kolekcije;
12. isti evaluator za baseline i Qdrant metode;
13. merljivu latency i quality osnovu za kasniji chatbot.

## 4. Režim pokretanja

Qdrant ima više lokalnih načina rada. Ne treba ih mešati pod jednim nazivom.

### 4.1 Qdrant Client Local Mode

Python client može da radi:

```text
in-memory:  QdrantClient(":memory:")
persistent: QdrantClient(path="...")
```

Prednosti:

- nema Docker servisa;
- nema otvorenog porta;
- jednostavan studentski setup;
- isti osnovni client interfejs kao server;
- pogodno za testove i Colab;
- 5.565 pointova je mali corpus za ovaj način rada.

Ograničenja koja treba proveriti compatibility spike-om:

- potpuna podrška za aktuelni Query API;
- podrška za sparse vectors, `Modifier.IDF`, RRF `k` i weighted RRF;
- ponašanje payload indexa u local implementaciji;
- podrška za collection aliases i snapshots;
- konkurentni pristup istoj persistent putanji;
- razlike u rezultatima ili filter semantici u odnosu na server.

Qdrant zvanično pozicionira Python Local Mode za razvoj, prototipe i testove.
Zato ga koristimo u prvoj fazi, ali ne pretpostavljamo da svaka server
operacija postoji lokalno.

### 4.2 Lokalni Qdrant Server kroz Docker

Prednosti:

- puna server implementacija;
- REST i gRPC;
- dashboard;
- jasnije testiranje server-only mogućnosti;
- bolji put ka zasebnom backend servisu;
- snapshots i aliases su prirodniji operativni model.

Nedostaci za trenutnu fazu:

- Docker postaje obavezan;
- servis i port moraju da se pokreću;
- dodatna infrastruktura za lokalnog korisnika;
- podrazumevani lokalni server nema autentikaciju i ne sme se nepotrebno
  izlagati mreži.

Docker server treba da bude opcioni compatibility profil, ne obavezan prvi
korak.

### 4.3 Qdrant Edge

Qdrant Edge je noviji embedded proizvod sa on-device vektorima i BM25
podrškom. Ne treba ga automatski poistovetiti sa Python Client Local Mode-om.
Može biti kasnija opcija ako želimo formalni embedded runtime ili kompatibilne
snapshotove, ali nije potreban za sadašnji corpus.

### 4.4 Preporuka

Početni redosled:

1. persistent Python Client Local Mode;
2. isti integracioni testovi protiv lokalnog Docker servera, samo ako je
   potreban server parity;
3. Cloud ili Edge samo ako se kasnije promeni deployment zahtev.

## 5. Lokacija baze na disku

Repository se nalazi pod OneDrive putanjom. Aktivna baza može imati česte male
upise i file lockove, pa sync klijent potencijalno može uticati na latenciju ili
pouzdanost.

Treba podržati konfigurabilnu putanju, umesto hardkodovanja.

Opcije:

### Repo-local

```text
data/processed/qdrant/
```

Prednosti:

- najjednostavniji studentski setup;
- svi generated artefakti su na jednom mestu;
- postojeći `.gitignore` već ignoriše `data/processed/`.

Rizik:

- folder je u OneDrive sinhronizovanoj putanji.

### Lokalni application-data folder

Primer koncepta:

```text
%LOCALAPPDATA%/BankScope/qdrant/
```

Prednosti:

- nije pod Gitom niti OneDrive-om;
- bolji za aktivnu persistent bazu;
- bliže realnom lokalnom application storage-u.

Nedostatak:

- manje je prenosivo između mašina;
- korisnik mora znati gde je baza;
- testovi i dokumentacija moraju koristiti konfiguraciju, ne relativnu
  putanju.

### Odluka koju treba doneti

Preporuka je:

- repo-local putanja za probe i demonstraciju;
- konfiguraciona promenljiva za nesinhronizovanu putanju kod redovne lokalne
  upotrebe;
- privremeni test folder za testove;
- nikada dve instance nad istom persistent putanjom bez eksplicitno proverene
  podrške.

## 6. Predložena struktura fajlova

```text
config/
  retrieval.yaml

data/processed/
  chunks.jsonl
  tables.jsonl
  manifest.json
  embeddings.npz
  qdrant/                         generated local DB
  qdrant_manifest.json           generated DB provenance

src/bankscope/vectorstore/
  __init__.py
  config.py                       validated Qdrant configuration
  point_mapping.py                corpus record -> point/payload
  sparse_encoder.py               document/query sparse encoding
  collection.py                   create/open/validate lifecycle
  indexer.py                      batched import
  qdrant_retriever.py             dense/sparse/hybrid queries
  manifest.py                     provenance and compatibility checks

src/bankscope/retrieval/
  evidence.py                     result normalization and table hydration

scripts/
  index_qdrant.py                 build/validate/info operations
  search.py                       backend selection during migration
  evaluate.py                     Qdrant methods added to evaluator

tests/
  test_qdrant_config.py
  test_qdrant_mapping.py
  test_qdrant_manifest.py
  test_qdrant_indexer.py
  test_qdrant_retriever.py
  test_qdrant_filters.py
  test_qdrant_persistence.py
  test_qdrant_evaluation.py
```

Ne treba napraviti sve module odjednom. Minimalni redosled je: config, mapping,
collection, indexer, retriever, evaluator.

## 7. Collection naming i verzionisanje

### 7.1 Fizički naziv

Predloženi obrazac:

```text
bankscope_retrieval_v1_<short-corpus-hash>
```

Prednosti:

- kolekcija je vezana za schema i corpus verziju;
- stara i nova verzija mogu privremeno koegzistirati;
- nema tihog prepisivanja prihvaćenog indeksa.

Za jednostavniju prvu iteraciju može se koristiti
`bankscope_retrieval_v1`, ali manifest i `--overwrite` zaštita tada postaju
obavezni.

### 7.2 Logical alias

Server Qdrant podržava atomsku zamenu collection aliasa. Potencijalni logical
naziv je:

```text
bankscope_active
```

Alias omogućava:

1. napraviti novu kolekciju;
2. potpuno je validirati;
3. atomski prebaciti alias;
4. zadržati staru kolekciju za rollback;
5. obrisati staru tek posle prihvatanja.

Local Mode podršku za aliases treba proveriti. Ako nije podržana, aktivni
collection name se menja u konfiguraciji tek posle evaluacije.

### 7.3 Collection metadata

Novije Qdrant verzije podržavaju collection metadata. Ona može sadržati mali
podskup provenance podataka, ali ne zamenjuje lokalni manifest jer:

- local/server podrška može da se razlikuje;
- manifest treba proveriti pre otvaranja baze;
- manifest se koristi i kada kolekcija nije dostupna.

## 8. Point model

Svaki zapis iz `chunks.jsonl` postaje tačno jedan Qdrant point.

```text
Qdrant point
  id
  vectors
    dense
    bm25
  payload
    BankScope identity
    filters
    provenance
    retrieval text
```

Razlog za jedan point po retrieval zapisu:

- dense i sparse predstavljaju isti retrieval kandidat;
- RRF ih prirodno spaja po istom point ID-ju;
- payload i filteri postoje samo jednom;
- target evidence identitet ostaje stabilan;
- text i table zapis imaju isti retrieval interfejs.

## 9. Point ID strategija

Qdrant point ID podržava 64-bit integer ili UUID. Postojeći `record_id` je
string sa namespace prefiksom i SHA digestom i ne treba pretpostaviti da je
validan sirov point ID.

Preporučena strategija:

- deterministički UUIDv5;
- fiksni BankScope namespace UUID;
- input je kompletan originalni `record_id`;
- originalni `record_id` ostaje u payloadu.

Obavezne provere:

- 5.565 različitih ulaza daju 5.565 različitih UUID vrednosti;
- isto indeksiranje daje iste UUID vrednosti;
- batch redosled ne menja mapiranje;
- reverse lookup je moguć preko payload `record_id`;
- point ID nikada ne postaje qrel/citation ID.

`target_chunk_id` ostaje evidence/qrel identitet. `record_id` ostaje retrieval
record identitet. Qdrant point ID je samo database identitet.

## 10. Named vector schema

### 10.1 Dense vector

Radni naziv:

```text
dense_qwen
```

Ugovor:

- size: 1024;
- distance: cosine;
- datatype: postojeći `float32` input;
- source: `embeddings.npz`;
- normalized input: da;
- model i revision: iz NPZ arhive, ne iz hardkodovane pretpostavke.

Qdrant cosine upload normalizuje vektore za efikasan dot-product search. Ipak,
BankScope pre uploada zadržava svoju proveru da su vektori konačni, nenulti i
pravilne dimenzije.

### 10.2 Sparse vector

Radni naziv:

```text
bm25
```

Ugovor za prvi eksperiment:

- encoder/model: `Qdrant/BM25` ili aktuelni zvanični ekvivalent potvrđen
  compatibility probe-om;
- input: isti `embedding_text` koji koristi dense i postojeći BM25S;
- collection sparse params uključuju `Modifier.IDF`;
- document i query obrada koriste identične model opcije;
- sparse indices su jedinstveni;
- sparse values su konačni;
- model/encoder verzija i opcije se upisuju u manifest.

### 10.3 Zašto named vectors

Named vectors omogućavaju:

- dense-only upit;
- sparse-only upit;
- native hybrid upit;
- kasnije dodavanje druge dense ili sparse reprezentacije;
- postepenu migraciju modela;
- jasno razlikovanje scoring prostora.

## 11. Qdrant/BM25 odluke

Qdrant BM25 sparse search nije automatski isti kao postojeći BM25S.

### 11.1 Trenutni BM25S ugovor

Aktivni kod koristi:

- NFKC normalizaciju;
- normalizaciju crtica;
- uklanjanje zareza između cifara;
- namenski regex koji čuva finansijske tokene i `%`;
- lowercase;
- bez stopword uklanjanja;
- bez stemminga;
- BM25S `lucene` metod.

### 11.2 Qdrant/BM25 default

Aktuelna Qdrant dokumentacija navodi da BM25 podrazumevano koristi:

- English language processing;
- English stemming;
- English stopword removal;
- word tokenizer;
- IDF statistiku iz shard/collection podataka.

To može biti bolje ili lošije za SEC pitanja, ali je promena retrieval metode.

### 11.3 Obavezni sparse eksperimenti

#### S0: postojeći BM25S

Kontrolni rezultat.

#### S1: Qdrant/BM25 default English

Meri standardnu i najjednostavniju Qdrant varijantu.

#### S2: Qdrant/BM25 konfiguracija bliža BM25S-u

Ako izabrani local inference put podržava potrebne opcije:

- isključen stemming;
- prazna stopword lista;
- word tokenizer;
- eksplicitno definisana normalizacija inputa pre encodera.

FastEmbed i server inference ne moraju podržavati identičan skup opcija.
Compatibility probe mora utvrditi šta zaista radi lokalno.

#### S3: learned sparse

SPLADE ili miniCOIL razmatrati samo ako:

- S1/S2 ne poprave alias ili lexical failure slučajeve;
- dodatna latencija je prihvatljiva;
- evaluacija pokaže dobitak;
- model i revision se mogu reproducibilno pinovati.

### 11.4 Izbor

Ne bira se prema prosečnom score-u, jer score skale nisu uporedive. Bira se
prema Hit@k, recall, MRR, evidence-group pokrivenosti, per-query failure diff-u
i latenciji.

## 12. Payload schema

Payload treba da bude dovoljno bogat za filtere, hydration i citate, ali ne
treba nekontrolisano kopirati sve iz corpusa.

### 12.1 Predložena obavezna polja

| Polje | Tip | Namena | Payload index |
|---|---|---|---|
| `record_id` | keyword/string | originalni retrieval ID | ne u prvoj fazi |
| `target_chunk_id` | keyword/string | qrel/evidence ID | opciono keyword |
| `record_type` | keyword | `text` ili `table` | da |
| `ticker` | keyword | bank filter | da |
| `embedding_text` | string | audit/debug i sparse source | ne |
| `document` | string | narrative evidence ili table opis | ne |
| `corpus_version` | keyword | schema/provenance kontrola | ne |

### 12.2 Uslovno obavezna polja

| Polje | Uslov | Namena | Payload index |
|---|---|---|---|
| `table_id` | table record | hydration cele tabele | opciono keyword |
| `accession_number` | kada postoji | filing provenance/filter | samo ako se filtrira |
| `report_year` | kada je pouzdano izveden | year filter | integer ako se koristi |
| `report_date` | kada postoji | provenance/range | datetime samo ako se koristi |
| `page` | kada postoji | citation | ne |
| `section_path` | kada postoji | citation/context | ne |
| `source_url` | kada postoji | reference | ne |

### 12.3 Polja koja ne treba izmišljati

- bank year ako se ne može pouzdano dobiti iz filing metadata;
- table title ako je samo heuristički generisan, bez oznake porekla;
- answer value ili units izdvojeni LLM-om;
- relevance score u payloadu;
- qrel status;
- korisnički query podaci.

### 12.4 Payload indexes

U prvoj fazi kreirati samo indekse koji se koriste u query filterima:

- `ticker`: keyword;
- `record_type`: keyword;
- `report_year`: integer, samo ako je polje pouzdano i tool ga koristi;
- eventualno `accession_number`: keyword, ako dokument lookup postane zahtev.

Qdrant preporučuje kreiranje payload indexa pre uploada pointova. To treba
poštovati čak i kod malog corpusa, jer schema ostaje pravilna za kasniji rast.

Ne treba indeksirati `embedding_text`, `document`, `source_url`, page ili svaki
provenance string bez konkretnog filter use-case-a.

## 13. Evidence storage i hydration

### 13.1 Narrative text

Za text point payload može sadržati narrative `document`, jer je to već
ograničen chunk i direktan dokaz.

Alternativa je samo čuvanje ID-ja i hydration iz `chunks.jsonl`. Prednost
payload kopije je jednostavniji retrieval; prednost hydration-a je jedan izvor
istine. Odluku treba potvrditi merenjem payload veličine, ali corpus je dovoljno
mali da je payload kopija razumna.

### 13.2 Table record

Table point payload sadrži:

- table description kao `document` ili `embedding_text`;
- `table_id`;
- metadata potrebne za citat.

Ne sadrži celu Markdown tabelu kao vector-search tekst.

Posle hita:

```text
Qdrant table point
  -> table_id
  -> tables_by_id lookup
  -> complete Markdown table
  -> evidence returned to caller
```

### 13.3 Table lookup

`tables.jsonl` ima samo 1.783 zapisa i može se učitati jednom po procesu u mapu
`table_id -> table`. Loader mora proveriti:

- jedinstvene table ID-jeve;
- da svaki table retrieval point ima odgovarajuću tabelu;
- da svaka vraćena tabela ima dokument i provenance;
- da Qdrant score nikada ne zameni evidence sadržaj.

## 14. Collection i database manifest

`qdrant_manifest.json` je obavezan čak i ako Qdrant podržava collection
metadata.

Predloženi sadržaj:

```text
schema_version
created_at_utc
deployment_mode
qdrant_path_or_url_kind
collection_name
logical_alias
point_count
text_point_count
table_point_count
chunks_sha256
tables_sha256
corpus_manifest_sha256
embeddings_sha256
dense_model_name
dense_model_revision
dense_dimensions
dense_distance
sparse_model_name
sparse_model_revision_or_package_version
sparse_options
sparse_modifier
rrf_configuration
payload_schema
payload_indexes
package_versions
build_parameters
```

Ne treba upisivati API key, apsolutne privatne putanje ili tajne.

### 14.1 Open-time validacija

Pre search-a proveriti:

1. manifest postoji;
2. schema version je podržan;
3. aktivni source hash-evi odgovaraju manifestu;
4. dense archive hash odgovara;
5. collection postoji;
6. collection vector schema odgovara;
7. exact count API vraća 5.565;
8. svi potrebni payload indexi postoje ili je odsustvo eksplicitno dozvoljeno;
9. sparse model/opcije odgovaraju query encoderu;
10. nema nedovršenog build statusa.

Ako bilo šta ne odgovara, search mora da stane sa jasnom porukom kako obnoviti
bazu.

### 14.2 Build status

Da bi se razlikovala kompletna od prekinute izgradnje:

- manifest se prvo piše kao `building` ili u privremeni fajl;
- posle svih provera postaje `ready`;
- search prihvata samo `ready`;
- neuspešna izgradnja ne prebacuje logical alias/config na novu kolekciju.

## 15. Dense import plan

### 15.1 Input provere

- NPZ ima sva obavezna polja;
- shape je `(5565, 1024)`;
- dtype je `float32` ili se kontrolisano konvertuje;
- svi brojevi su konačni;
- nema zero vectora;
- norme su očekivane;
- `record_ids` su jedinstveni;
- redosled tačno odgovara `chunks.jsonl`;
- `input_sha256` odgovara `chunks.jsonl`;
- model name/revision nisu promenjeni.

### 15.2 Sparse generisanje

Sparse document encoding radi nad `embedding_text`. Treba da bude streaming ili
batch proces, tako da se ne duplira nepotrebno ceo corpus u memoriji.

Pre uploada proveriti:

- broj sparse rezultata odgovara broju zapisa;
- svaki rezultat ima usklađene indices i values;
- nema dupliranih indices;
- nema NaN/Inf vrednosti;
- prazan sparse vector se eksplicitno obrađuje ili odbija;
- document i query encoder koriste isti model i opcije.

### 15.3 Redosled izgradnje

1. validirati source artefakte;
2. odabrati nov versioned collection name;
3. kreirati collection sa dense i sparse named vectorima;
4. kreirati payload indexes;
5. generisati point ID-jeve i payload;
6. generisati sparse vectors;
7. spojiti postojeći dense i novi sparse vector u point;
8. uploadovati batch-eve;
9. sačekati završetak upisa/optimizacije gde je relevantno;
10. proveriti exact count;
11. scroll/inspect deterministički uzorak;
12. izvršiti integrity testove;
13. upisati finalni manifest;
14. tek posle evaluacije promeniti aktivnu kolekciju.

### 15.4 Batch upload

Zvanična Qdrant dokumentacija preporučuje batch upload umesto point-by-point
upisa i navodi 64-256 pointova kao praktičan početni opseg za server.

Za naš corpus:

- početni batch kandidat: 128;
- bez agresivne paralelizacije u Local Mode-u;
- server profil može kasnije probati 2-4 workera;
- svaki batch mora biti idempotentan;
- retries ne smeju kreirati duplikate jer point ID ostaje isti;
- `wait`/completion semantiku treba potvrditi kroz client API.

## 16. Dense search režim

### 16.1 Exact search kao parity režim

Sa 5.565 vektora exact search je realan početni izbor. Njegova svrha je:

- poređenje sa postojećim NumPy cosine rankingom;
- determinističniji rezultat;
- izolovanje point/payload grešaka;
- baseline pre HNSW podešavanja.

Query mora:

- koristiti `dense_qwen` named vector;
- slati normalizovani Qwen query vector;
- tražiti samo potreban payload;
- ne vraćati same vector values;
- primeniti isti ticker/record-type filter kao baseline;
- koristiti dovoljan `limit` za evaluator.

### 16.2 HNSW kao opcioni speed režim

HNSW je approximate i može promeniti top rezultate. Uvodi se samo ako merenje
pokaže da exact search predstavlja relevantan deo ukupne latencije.

Parametri za kasniju evaluaciju:

- `m`;
- `ef_construct`;
- query `hnsw_ef`;
- `full_scan_threshold`;
- payload filter ponašanje;
- memory i index build vreme.

Ne postavljati vrednosti samo prema generičkim primerima. Naš corpus je mali i
filteri po banci dodatno smanjuju candidate set.

### 16.3 Quantization

Quantization trenutno nije potrebna:

- dense matrica je oko nekoliko desetina MB;
- nema RAM pritiska;
- može promeniti recall;
- dodaje konfiguraciju i rescoring odluke.

Razmatrati scalar/binary/product quantization tek kod većeg corpusa ili
izmerenog memory problema.

### 16.4 On-disk vectors

Dense vectors mogu ostati u memoriji. `on_disk`/memmap nema jasnu korist za
5.565 pointova i može dodati disk IO. Razmatrati samo ako corpus znatno poraste.

## 17. Sparse search režim

Qdrant sparse search koristi sparse indices/values i po pravilu radi exact
pretragu nad preklapajućim dimenzijama.

Query mora:

- koristiti `bm25` named vector;
- koristiti isti encoder i opcije kao indexing;
- primeniti iste filters;
- vraćati isti tip normalized resulta kao dense search;
- zabeležiti raw sparse score za dijagnostiku;
- ne porediti raw sparse score direktno sa cosine score-om.

Sparse score nije answerability probability i ne treba uvoditi proizvoljan
globalni threshold bez evaluacije.

## 18. Native Qdrant RRF

### 18.1 Query struktura

Native hybrid query treba da ima:

```text
prefetch 1: sparse query, using bm25, limit=candidate_k
prefetch 2: dense query, using dense_qwen, limit=candidate_k
main query: RRF
main limit: requested result count
filter: isti semantički filter za oba retrieval signala
with_payload: required fields
with_vectors: false
```

Početni parametri iz aktivnog sistema:

- `candidate_k=30`;
- final `limit=10` za evaluaciju;
- jednaki sparse i dense weights;
- bez formula boostova;
- bez rerankera.

### 18.2 Važna razlika u rank indeksiranju

Postojeći BankScope RRF koristi rangove od 1:

```text
score = 1 / (60 + rank_1_based)
```

Qdrant dokumentacija definiše prvi rezultat kao rank 0:

```text
score = 1 / (k + rank_0_based)
```

Zato su potrebna dva eksplicitna eksperimenta:

#### R1: parity konstanta

```text
Qdrant k=61
```

Ovo matematički odgovara postojećem doprinosu ako su ulazni rangovi isti:

```text
1 / (60 + rank_1_based)
=
1 / (61 + rank_0_based)
```

#### R2: nominalna konstanta

```text
Qdrant k=60
```

Ovo prati uobičajeni naziv postojećeg parametra, ali nije potpuno isti scoring
ugovor zbog zero-based ranka.

Rezultati se ne smeju pomešati pod jednim nazivom `rrf_k=60`.

### 18.3 Weighted RRF

Aktuelni Qdrant podržava weights po prefetch listi. Ne koristiti weighted RRF
u prvom prihvaćenom rezultatu zato što:

- imamo samo 28 odgovorivih pitanja;
- tuning i evaluacija na istim pitanjima daju optimističan rezultat;
- menja se i sparse retriever, pa je teško izolovati uzrok.

Ako ga kasnije koristimo:

- unapred zamrznuti train/validation split;
- težine povezati sa redosledom prefetch upita;
- sačuvati sve probane konfiguracije;
- rezultat na validation skupu prijaviti odvojeno;
- ponoviti tuning kada se promeni corpus/model/chunking.

### 18.4 DBSF i formula query

Qdrant podržava DBSF i formula queries. Oni su korisni za:

- normalizaciju score distribucija;
- recency ili business-rule boostove;
- višestepeno pretraživanje.

Za sada ih ne koristiti:

- naš baseline je rank-based RRF;
- SEC filing godina treba da bude filter/query intent, ne proizvoljni recency
  boost;
- dodatne formule otežavaju objašnjenje rezultata;
- prvo treba dobiti čist dense+sparse rezultat.

## 19. Filter plan

### 19.1 Ticker

- exact keyword match;
- normalizovan uppercase pre query-ja;
- tool/aplikacija bira dozvoljeni ticker;
- ne koristiti full-text match;
- testirati svaki od deset tickera.

### 19.2 Record type

- exact keyword;
- dozvoljene vrednosti `text`, `table`;
- `any` znači da se filter ne šalje, nije payload vrednost;
- testirati da table-only nikada ne vraća text point i obrnuto.

### 19.3 Report year/date

Uvesti tek nakon provere corpus metadata:

- godina mora poticati iz filing/report metadata;
- ne iz LLM zaključka;
- koristiti integer index za godinu;
- datetime index samo ako range query stvarno treba;
- ako corpus sadrži samo jedan report period po banci, filter možda nije
  potreban u prvoj verziji.

### 19.4 Accession number

Koristan za document lookup ili audit. Ne mora biti indeksiran dok se ne koristi
kao query filter.

### 19.5 Kombinovanje filtera

BankScope standardna semantika:

```text
must:
  ticker, kada je naveden
  record_type, kada nije any
  report_year, kada je pouzdan i naveden
```

`should` i `must_not` se ne uvode bez konkretnog use-case-a. Za više tickera
koristiti `match any` ili odvojene upite, zavisno od cross-bank strategije.

## 20. Result normalizacija

Qdrant response ne treba direktno vraćati LangChainu, LangGraphu ili UI-ju.
Mapira se na BankScope retrieval rezultat:

```text
record_id
target_chunk_id
record_type
ticker
document/evidence
metadata
retrieval_method
rank
score
dense_rank, kada postoji
sparse_rank, kada postoji
point_id, samo development dijagnostika
```

Za Qdrant-native RRF API možda ne vraća pojedinačne dense i sparse rankove u
finalnom resultu. Ako su potrebni za evaluaciju/dijagnostiku, opcije su:

1. pokrenuti dense i sparse upite odvojeno uz native hybrid;
2. rekonstruisati rangove iz sačuvanih prefetch rezultata;
3. u production search-u izostaviti pojedinačne rangove, ali ih uključiti u
   evaluator režimu.

Ne treba izmišljati rank vrednosti koje Qdrant nije vratio.

## 21. Search API na nivou projekta

Qdrant retriever treba da podrži najmanje:

```text
search_dense(query_vector, filters, limit)
search_sparse(query_text, filters, limit)
search_hybrid(query_text, query_vector, filters, limit, candidate_k, rrf)
```

Query encoder ostaje van Qdrant storage klase:

- dense Qwen encoder daje query vector;
- sparse encoder daje text/document ili sparse vector, zavisno od potvrđenog
  client puta;
- Qdrant klasa izvršava database query;
- evidence sloj hydrate-uje rezultat.

Ovo omogućava da evaluator testira database bez LLM-a i da se query encoder
kešira jednom po procesu.

## 22. Index lifecycle komande

Predložene operacije jednog CLI-ja:

```text
index_qdrant build
index_qdrant validate
index_qdrant info
index_qdrant compare-manifest
```

Opcioni kasniji podkomandni tokovi:

```text
index_qdrant snapshot
index_qdrant restore
index_qdrant switch-alias
index_qdrant delete
```

Destruktivne operacije moraju:

- zahtevati eksplicitno ime/putanju;
- proveriti da je cilj unutar konfigurisanog Qdrant storage-a;
- ne prihvatati repository root, home ili praznu putanju;
- prikazati collection name i point count;
- koristiti `--yes` samo kada je korisnik jasno odabrao operaciju;
- nikada ne brisati kanonske JSONL/NPZ artefakte.

## 23. Idempotency i update strategija

### Full rebuild kao početna strategija

Corpus se trenutno gradi batch putem i ima samo 5.565 zapisa. Najjasniji
početni model je full rebuild nove versioned kolekcije.

Prednosti:

- izbegava stale pointove;
- jednostavan provenance;
- lako poređenje i rollback;
- nema komplikovanog diff algoritma.

### Incremental update kao kasnija mogućnost

Qdrant podržava upsert i delete, ali incremental ingestion zahteva:

- tačno prepoznavanje novih/izmenjenih/obrisanih record ID-jeva;
- re-embedovanje izmenjenih dense zapisa;
- ponovno sparse kodiranje;
- IDF/statistics posledice;
- manifest sa više source generacija;
- test da nema orphan pointova.

Ne uvoditi dok download/corpus pipeline nema jasan incremental zahtev.

## 24. Persistence, backup i recovery

### 24.1 Izvorni recovery plan

Primarni recovery je rebuild iz:

- `chunks.jsonl`;
- `tables.jsonl`;
- `manifest.json`;
- `embeddings.npz`;
- pinovanog sparse encodera i konfiguracije.

Zato Qdrant storage nije jedina kopija podataka.

### 24.2 Local Mode backup

Ne pretpostavljati da server snapshot API radi u Python Local Mode-u.
Compatibility probe treba da potvrdi podršku.

Ako snapshots nisu podržani:

- zatvoriti client/proces pre kopiranja storage foldera;
- kopirati ceo folder kao jednu celinu;
- kopirati odgovarajući manifest;
- testirati otvaranje kopije;
- ne koristiti live OneDrive kopiju kao dokaz konzistentnog backupa.

Pošto je rebuild jeftin, folder copy backup nije obavezan u prvoj fazi.

### 24.3 Server snapshots

Qdrant server snapshot sadrži collection konfiguraciju, pointove i payload.
Collection aliases nisu uključeni i moraju se obnoviti zasebno.

Snapshots postaju relevantni ako:

- Docker/server postane redovan način rada;
- sparse build postane skup;
- želimo prenos kolekcije između mašina;
- baza više nije lako obnovljiva.

## 25. Concurrency i process model

Za prvu lokalnu verziju:

- jedan Python proces poseduje persistent Qdrant Local client;
- model i client se učitavaju jednom;
- ne otvarati isti path iz više procesa;
- ne isključivati SQLite/thread safety provere bez jasnog razloga;
- indexing i search ne raditi paralelno nad istom versioned kolekcijom;
- nova kolekcija se gradi odvojeno.

Ako kasnije više procesa mora deliti bazu, preći na lokalni Qdrant server umesto
forsiranja Local Mode thread/process ponašanja.

## 26. Security

### Local Mode

- nema mrežnog porta;
- nema API key-a;
- storage path ne sadrži tajne;
- SEC dokumenti su javni, ali budući korisnički upiti nisu deo baze;
- Qdrant folder ostaje Git ignored.

### Docker/server

- bindovati samo na localhost za lokalni razvoj;
- ne izlagati podrazumevani neautentikovani port mreži;
- autentikaciju/TLS razmatrati tek za udaljeni pristup;
- API key nikada ne upisivati u repository config;
- snapshot folder tretirati kao podatke, ne source code.

## 27. Test plan

### 27.1 Unit testovi

- stable UUID mapping;
- payload required/optional polja;
- dense vector validacija;
- sparse vector validacija;
- manifest serialization i validation;
- filter builder;
- Qdrant response -> BankScope result mapping;
- table hydration;
- RRF config transform i jasno razlikovanje Qdrant `k=60/61`.

### 27.2 Local integration test

Koristiti privremeni persistent folder i mali fixture corpus:

- create collection;
- create payload indexes;
- insert text i table point;
- close client;
- reopen client;
- exact dense query;
- sparse query;
- hybrid RRF query;
- ticker filter;
- record-type filter;
- count API;
- scroll sa payload selectorom;
- stale manifest rejection;
- table hydration.

### 27.3 Local/server parity test

Opcioni test nad istim fixture-om:

- Local Mode i Docker server;
- isti vectors/payload;
- isti dense/sparse/hybrid upiti;
- isti filteri;
- poređenje top ID-jeva i score semantike;
- zabeležene nepodržane local operacije.

### 27.4 Corpus integrity test

- exact point count 5.565;
- svi record ID-jevi jedinstveni;
- svi source IDs postoje;
- 1.556 table pointova;
- svi table ID-jevi postoje u table store-u;
- ticker raspodela odgovara corpus-u;
- record-type raspodela odgovara manifestu;
- nasumični i deterministički uzorak payload/vectors.

## 28. Retrieval evaluation plan

Evaluator treba proširiti bez brisanja postojećih metoda.

Predložene metode:

```text
numpy_dense
bm25s
baseline_hybrid
qdrant_dense_exact
qdrant_bm25_default
qdrant_bm25_compatible
qdrant_hybrid_k61
qdrant_hybrid_k60
qdrant_hybrid_weighted       kasnije
qdrant_dense_hnsw            kasnije
```

Za svaku metodu čuvati:

- Hit@1/3/5/10;
- mean Recall@1/3/5/10;
- MRR@10;
- evidence-group recall i complete hit rate;
- per-query top rezultate;
- filtere;
- model/config provenance;
- candidate limit;
- final limit;
- Qdrant deployment mode i verziju.

### 28.1 Dense parity gate

Qdrant exact dense mora najmanje da zadovolji:

- nema unknown/missing pointova;
- nema gubitka relevantnih rezultata zbog mappinga;
- agregatne dense metrike nisu lošije;
- top-10 razlike su samo objašnjive tie/normalization razlike;
- table hydration daje isti evidence ID i sadržaj.

### 28.2 Sparse gate

Poređenje sa BM25S:

- Hit@5 baseline 25/28;
- Hit@10 baseline 26/28;
- MRR@10 baseline 0,562;
- pregled tri alias pitanja;
- pregled table exact-value i parser metadata pitanja;
- pregled pitanja koja sparse metoda izgubi ili dobije.

### 28.3 Hybrid gate

Poređenje sa postojećim hybridom:

- Hit@5 baseline 25/28;
- Hit@10 baseline 26/28;
- MRR@10 baseline 0,614;
- Mean Recall@10 baseline 0,855;
- cross-bank complete group @10 baseline 2/3;
- nema skrivene regresije cele kategorije pitanja.

Nova metoda nije prihvaćena samo zato što je jedna agregatna metrika viša.

## 29. Latency benchmark plan

Vector database može ubrzati neighbor search, ali ne rešava automatski query
encoding ili aplikacioni startup.

Meriti odvojeno:

### Startup

- import modula;
- otvaranje Qdrant Local baze;
- manifest validation;
- učitavanje table lookup-a;
- učitavanje Qwen query encodera;
- učitavanje sparse encodera.

### Per-query

- dense query encoding;
- sparse query encoding;
- dense database query;
- sparse database query;
- native hybrid database query;
- payload transfer/mapping;
- evidence hydration;
- ukupan retrieval.

### Scenariji

- cold first query;
- warm single query;
- 28 evaluation queries batch;
- ticker-filtered query;
- unfiltered query;
- table-only query;
- cross-bank više upita;
- Local Mode;
- Docker server samo ako ga podržimo.

### Metodologija

- ista mašina i power profil;
- više warmup upita;
- više ponavljanja;
- median i p95, ne samo prosečno vreme;
- database timing bez LLM API-ja;
- vector values se ne vraćaju u rezultatima;
- zabeležiti model cache stanje.

Ne postavljati proizvoljan latency target pre baseline merenja.

## 30. Qdrant mogućnosti koje možemo kasnije iskoristiti

### 30.1 Weighted RRF

Ako dense i sparse imaju različitu snagu, Qdrant može zadati weight po
prefetch-u. Koristiti samo sa development/validation splitom.

### 30.2 DBSF

Alternativa RRF-u kada želimo da normalizovane score distribucije utiču na
fusion. Potrebna zasebna evaluacija.

### 30.3 Learned sparse vectors

SPLADE ili miniCOIL mogu pomoći terminima i aliasima. Dodatni model i latencija
moraju biti opravdani.

### 30.4 Multi-vector ili ColBERT reranking

Qdrant može čuvati dodatne named/multivectors za late interaction ili rescoring.
To je potencijalna kasnija reranker arhitektura, ne deo početne migracije.

### 30.5 Grouping

`query_points_groups` može grupisati rezultate po payload polju. Moguće
primene:

- ograničiti više rezultata iz iste tabele/dokumenta;
- dobiti raznovrsniji filing coverage;
- grupisati po tickeru u dijagnostici.

Grouping nije zamena za query decomposition kada pitanje traži tačno određen
dokaz za svaku banku.

### 30.6 Batch i paralelni query

Cross-bank decomposition može slati više filtriranih upita. Async client ili
batch query može smanjiti čekanje, ali tek nakon pravilnog single-query toka.

### 30.7 Full-text filteri i phrase match

Text payload index omogućava token/phrase filtere. Može pomoći document lookup
ili exact phrase funkciji, ali ne treba ga mešati sa BM25 rankingom bez jasnog
use-case-a.

### 30.8 Formula query

Može dodati recency ili metadata boost. Za SEC filing pitanja godina je
verovatnije hard filter ili deo query plana nego globalni recency boost.

### 30.9 Collection aliases

Omogućavaju versioned rebuild i atomski switch na serveru. Korisno kada baza
postane dugotrajni servis.

### 30.10 Snapshots

Korisni za server backup i prenos. Local Mode podršku treba proveriti; rebuild
ostaje primarni plan.

### 30.11 Quantization i on-disk storage

Korisno za znatno veći corpus. Trenutno nije potrebno.

### 30.12 Strict mode

Server strict mode može sprečiti neindeksirane filtere, prevelike batch-eve i
neefikasne upite. Razmotriti kada pređemo sa embedded Local Mode-a na servis.

### 30.13 Payload-based multitenancy

Nije potrebno za jednog lokalnog korisnika. Ako kasnije postoje različiti
corpus-i ili korisnici, payload partitioning je verovatno bolji od mnogo malih
kolekcija.

## 31. Mogućnosti koje ne treba koristiti sada

- Qdrant Cloud;
- distribuirani shardovi i replike;
- multitenancy;
- GPU indexing;
- quantization;
- custom recency/popularity formula;
- više dense modela;
- learned sparse pre BM25 baseline-a;
- server snapshots kao jedini backup;
- incremental update pipeline;
- automatsko collection brisanje;
- storage unutar Git istorije;
- full table Markdown kao embedding input;
- retrieval score kao answer confidence.

Razlog nije da su ove mogućnosti loše, već da trenutno ne rešavaju izmeren
problem.

## 32. Dependency plan

### Potrebno za Qdrant core

- `qdrant-client`;
- FastEmbed podrška ako se koristi lokalni `Qdrant/BM25` encoder;
- postojeći NumPy;
- postojeći Pydantic/pydantic-settings za konfiguraciju i manifest;
- postojeći Sentence Transformers za Qwen query embedding.

### Compatibility provere pre pinovanja

- Python 3.13 podrška;
- Qdrant Local persistent path;
- native sparse i RRF Query API;
- `Rrf(k=...)` podrška;
- weighted RRF dostupnost ako je želimo kasnije;
- `Modifier.IDF` u Local Mode-u;
- FastEmbed model download/cache;
- offline ponašanje posle prvog download-a;
- konflikt sa postojećim NumPy, Transformers i ONNX Runtime paketima;
- Windows i OneDrive file-lock ponašanje.

### Optional dependency grupa

Qdrant može prvo biti optional grupa, na primer konceptualno `.[qdrant]`, dok
postojeći baseline ostaje instalabilan. Nakon prihvatanja Qdranta možemo
odlučiti da li postaje core dependency.

Tačne verzije se dodaju tek posle probe-a; ovaj plan ne izmišlja kompatibilne
verzije.

## 33. Implementacione faze

### Faza Q0: potvrda otvorenih odluka

Potvrditi:

- repo-local ili application-data path;
- persistent Local Mode kao prvi cilj;
- `Qdrant/BM25` kao prvi sparse encoder;
- sparse default English i BM25S-like eksperimente;
- versioned collection naming;
- dense exact kao parity režim;
- kriterijume za prihvatanje regresije, ako je ima.

### Faza Q1: compatibility spike

Izolovano potvrditi:

- instalaciju na Pythonu 3.13;
- create/open persistent local DB;
- dense named vector;
- sparse named vector sa IDF modifierom;
- BM25 document i query encoding;
- native RRF sa eksplicitnim `k`;
- payload filter/index;
- close/reopen;
- list/count/scroll;
- šta od aliases/snapshots nije podržano lokalno.

Exit uslov: kratak zabeležen compatibility rezultat bez izmene aktivnog
retrievera.

### Faza Q2: config, mapping i manifest

Implementirati:

- validated config;
- stable point ID;
- payload mapping;
- collection schema;
- manifest schema;
- unit testove.

Exit uslov: fixture collection može deterministički da se obnovi i validira.

### Faza Q3: full corpus dense import

Importovati postojeće dense embeddinge, bez sparse/hybrid odluke u active kodu.

Exit uslov:

- 5.565 validnih pointova;
- svi integritet testovi;
- dense exact parity evaluation;
- latency poređenje NumPy/Qdrant.

### Faza Q4: Qdrant/BM25 sparse

Implementirati S1 i, ako je podržano, S2.

Exit uslov:

- sparse integrity;
- BM25S vs Qdrant per-query diff;
- zabeležen izbor sparse konfiguracije.

### Faza Q5: native RRF

Implementirati R1 (`k=61`) i R2 (`k=60`).

Exit uslov:

- puna 28-query evaluacija;
- cross-bank group metrike;
- latency;
- odluka da li Qdrant native hybrid prolazi quality gate.

### Faza Q6: stabilni retrieval servis

Uvesti BankScope result mapping, filtere i table hydration. Baseline i Qdrant
backend koegzistiraju tokom tranzicije.

Exit uslov: isti search/evaluate contract radi nad oba backenda.

### Faza Q7: prihvatanje i cleanup odluka

Ako Qdrant prođe:

- zabeležiti decision record;
- promeniti default backend;
- ažurirati README/data pipeline/roadmap;
- odlučiti da li BM25S ostaje fallback ili samo regression baseline;
- zadržati rebuild i validation komande;
- ne brisati stare artefakte dok rollback više nije potreban.

Ako ne prođe:

- zadržati Qdrant dense storage ako donosi operativnu korist i dense parity;
- zadržati BM25S/RRF u aplikaciji; ili
- odbaciti Qdrant migraciju bez skrivanja rezultata.

## 34. Definition of done

Qdrant faza je završena kada:

- persistent lokalna baza radi na podržanom Python okruženju;
- baza se potpuno gradi iz kanonskih artefakata;
- point i payload schema su dokumentovani i testirani;
- postojeći dense embedding se importuje bez re-embedovanja;
- sparse model i opcije su eksplicitno zabeleženi;
- native dense, sparse i RRF search rade;
- ticker i record-type filteri rade;
- cele tabele se vraćaju kroz `table_id` hydration;
- stale collection se odbija preko manifest ugovora;
- 5.565 pointova i source hash-evi su potvrđeni;
- dense parity je potvrđen;
- Qdrant sparse/hybrid je upoređen sa BM25S/hybrid baseline-om;
- latency je izmerena po komponentama;
- aktivni backend je promenjen samo uz decision record;
- generated baza i modeli nisu slučajno commitovani;
- dokumentacija objašnjava build, validate, search, evaluate i recovery.

## 35. Otvorene odluke koje plan ne pretpostavlja

1. Da li persistent baza ostaje unutar OneDrive repository-ja?
2. Koja tačna `qdrant-client`/FastEmbed verzija podržava potreban API na
   Pythonu 3.13?
3. Da li Local Mode podržava sve potrebne RRF i sparse opcije?
4. Da li FastEmbed put podržava BM25 opcije potrebne za S2?
5. Da li Qdrant default BM25 nadmašuje postojeći no-stem/no-stopword BM25S?
6. Da li `k=61` daje očekivani parity kada se promeni sparse rangiranje?
7. Da li exact dense search ima merljivu prednost ili regresiju u odnosu na
   NumPy?
8. Da li HNSW uopšte ima smisla na 5.565 pointova?
9. Da li aliases i snapshots rade u izabranom Local Mode-u?
10. Da li `langchain-qdrant` kasnije izlaže dovoljan native query nivo ili će
    aplikacija koristiti direktan Qdrant client?

Odgovori na ova pitanja dobijaju se kroz Q1-Q5, ne pretpostavkama.

## 36. Neposredni sledeći korak

Kada se odobri ovaj plan, prvi i jedini početni implementacioni zadatak treba
da bude Faza Q1: mali persistent Qdrant Local compatibility spike.

Spike ne treba da koristi ceo corpus niti da menja aktivni `search.py`. Dovoljna
su dva do četiri fixture pointa koja dokazuju:

- dense import;
- BM25 sparse encoding;
- native RRF sa `k=60` i `k=61`;
- payload filter;
- close/reopen;
- podržane i nepodržane Local Mode operacije.

Tek posle tog rezultata treba dodati dependencies i full-corpus indexing u
aktivni projekat.

## 37. Zvanične reference

- Qdrant Python client i persistent Local Mode:
  <https://github.com/qdrant/qdrant-client>
- Qdrant collections, named vectors, metadata i aliases:
  <https://qdrant.tech/documentation/manage-data/collections/>
- Qdrant payload:
  <https://qdrant.tech/documentation/concepts/payload/>
- Payload i vector indexing:
  <https://qdrant.tech/documentation/manage-data/indexing/>
- Filtering:
  <https://qdrant.tech/documentation/search/filtering/>
- Dense i sparse search:
  <https://qdrant.tech/documentation/search/>
- BM25 full-text/sparse search i text processing:
  <https://qdrant.tech/documentation/search/text-search/full-text-search/>
- Server-side BM25 inference:
  <https://qdrant.tech/documentation/inference/inference-bm25/>
- FastEmbed:
  <https://qdrant.tech/documentation/fastembed/>
- Hybrid Query API, RRF `k`, weighted RRF i DBSF:
  <https://qdrant.tech/documentation/search/hybrid-queries/>
- Qdrant fundamentals i exact/HNSW smernice:
  <https://qdrant.tech/documentation/faq/qdrant-fundamentals/>
- Bulk upload:
  <https://qdrant.tech/documentation/database-tutorials/bulk-upload/>
- Snapshots:
  <https://qdrant.tech/documentation/operations/snapshots/>
- Qdrant LangChain integracija, samo za kasniju granicu:
  <https://qdrant.tech/documentation/frameworks/langchain/>

