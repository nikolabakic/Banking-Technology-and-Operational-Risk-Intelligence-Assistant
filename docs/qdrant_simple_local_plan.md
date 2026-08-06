# Pojednostavljen plan: lokalni Qdrant retrieval

## 1. Cilj ove faze

Cilj je da postojeći retrieval smestimo u lokalnu Qdrant bazu kako bi naredni
koraci — retrieval tool, GPT-4o odgovaranje, citati i kasnije chatbot — bili
brži i jednostavniji.

U ovoj fazi ne menjamo corpus, ne pravimo nove dense embeddinge i ne
optimizujemo sistem posle svakog koraka. Implementiramo jednu razumnu verziju,
a zatim je jednom evaluiramo na postojećih 30 pitanja.

## 2. Odluke koje unapred prihvatamo

Koristimo:

- Qdrant u persistent Local Mode-u, bez servera i Dockera;
- jednu kolekciju `bankscope_retrieval`;
- postojeće Qwen dense embeddinge iz `embeddings.npz`;
- Qdrant/BM25 za sparse retrieval;
- Qdrant native RRF za spajanje dense i sparse rezultata;
- postojeći whole-table pristup: retrieval pronalazi tabelu, a LLM kasnije
  čita celu tabelu i izvlači odgovor;
- postojeći skup od 30 pitanja samo za završno poređenje.

Za sada ne uvodimo LangChain, LangGraph, reranker, Docker, cloud bazu,
snapshots, aliases, weighted fusion niti dodatne sparse modele. Ti dodaci imaju
smisla tek ako osnovni lokalni retrieval radi i ako postoji jasan problem koji
treba rešiti.

## 3. Ulazni fajlovi

Koristimo postojeće artefakte:

```text
data/processed/chunks.jsonl
data/processed/tables.jsonl
data/processed/embeddings.npz
data/processed/manifest.json
data/evaluation/queries.jsonl
```

Ne preuzimamo ponovo dokumente, ne pokrećemo parser i ne radimo novo GPU
embedovanje.

## 4. Lokalna baza

Baza se čuva u:

```text
data/processed/qdrant/
```

To je generisani lokalni artefakt i ne treba ga commitovati. Izvor istine
ostaju JSONL i NPZ fajlovi, tako da se baza po potrebi može ponovo izgraditi.

Kolekcija ima dva named vector polja:

```text
dense   Qwen, 1024 dimenzije, cosine distance
sparse  Qdrant/BM25
```

Svaki zapis iz `chunks.jsonl` postaje jedan Qdrant point. Payload sadrži samo
ono što je potrebno:

- `record_id` i `target_chunk_id` za retrieval, evaluaciju i citate;
- `record_type` (`text` ili `table`);
- `ticker` i naziv banke;
- `table_id`, kada je zapis tabela;
- dokument, filing tip, datum ili godina i accession number, kada postoje;
- naslov sekcije ili tabele i broj strane, kada postoje u izvornim metadata;
- putanju ili drugi identifikator izvornog dokumenta;
- tekst korišćen za retrieval;
- ostale postojeće source metadata koje su potrebne za proveru porekla dokaza.

Metadata imaju tri namene:

- **retrieval filteri sada:** `ticker` i `record_type`;
- **citati i LLM kontekst:** banka, dokument, datum ili godina, sekcija, tabela
  i stranica kada je dostupna;
- **mogući budući filteri:** filing tip, godina i accession number.

Sva dostupna metadata čuvamo u Qdrant payloadu, ali u prvoj verziji pravimo
filter indekse samo za `ticker` i `record_type`. Ostala polja služe za prikaz
izvora, citate i LLM kontekst; indeksiraćemo ih tek ako ih retrieval funkcija
zaista bude koristila za filtriranje.

## 5. Kako radi pretraga

Za svaki upit izvršavamo:

1. dense pretragu postojećim Qwen query embeddingom;
2. sparse pretragu Qdrant/BM25 modelom;
3. native RRF spajanje dve rang-liste;
4. vraćanje najboljih 10 rezultata.

Početna podešavanja su fiksna:

```text
dense candidates: 30
sparse candidates: 30
RRF: standardni Qdrant RRF
final results: 10
```

Ne radimo grid search i ne podešavamo parametre na istih 30 pitanja. Ako je
rezultat na kraju lošiji, prvo analiziramo promašena pitanja, pa tek onda
biramo jednu ciljanu izmenu.

Kada rezultat predstavlja tabelu, Qdrant vraća `table_id`, a aplikacija iz
`tables.jsonl` učitava celu Markdown tabelu. Opis tabele služi za pronalaženje;
originalna tabela služi kao dokaz i kasniji ulaz za LLM.

## 6. Minimalne izmene u projektu

Planirana struktura je:

```text
src/bankscope/retrieval/qdrant_retriever.py
scripts/build_qdrant.py
scripts/search.py          # proširen postojećim Qdrant backendom
scripts/evaluate.py        # koristi postojeća pitanja i metrike
tests/test_qdrant_retriever.py
```

`build_qdrant.py` pravi kolekciju, uvozi postojeće dense vektore, generiše
sparse vektore i upisuje payload.

`qdrant_retriever.py` otvara lokalnu bazu, izvršava dense, sparse ili hybrid
pretragu, primenjuje jednostavne filtere i učitava celu tabelu kada je potrebno.

Ne pravimo poseban framework, drugi evaluator niti LangChain wrapper u ovoj
fazi.

## 7. Redosled rada

### Korak 1 — lokalni tehnički smoke test

Na nekoliko probnih zapisa proveriti da Python okruženje može da kreira,
zatvori i ponovo otvori lokalnu Qdrant bazu sa dense, sparse i RRF pretragom.
Ovo nije evaluacija kvaliteta, već samo brza provera kompatibilnosti.

### Korak 2 — implementacija retrievala

Napraviti build skriptu i retriever, povezati postojeće embeddinge i whole-table
hydration, pa dodati Qdrant opciju postojećim `search.py` i `evaluate.py`.

### Korak 3 — izgradnja cele baze

Jednom ubaciti svih 5.565 retrieval zapisa. Na kraju build-a proveriti samo:

- da kolekcija ima 5.565 pointova;
- da svi dense vektori imaju 1.024 dimenzije;
- da nema duplih ID-jeva;
- da svaki table zapis pokazuje na postojeću celu tabelu.

Ove provere sprečavaju da evaluiramo tehnički neispravnu bazu. Ne pokrećemo
retrieval metrike između koraka.

### Korak 4 — jedna završna evaluacija

Tek kada cela putanja radi, pokrenuti svih 30 postojećih pitanja. Kao i do sada,
28 answerable pitanja ulazi u metrike, a ambiguous i unsupported ostaju
dijagnostička.

Uporediti:

| Metoda | Svrha |
|---|---|
| Postojeći BM25S | prihvaćeni lexical baseline |
| Postojeći dense | prihvaćeni semantic baseline |
| Postojeći hybrid | glavni trenutni baseline |
| Qdrant sparse | provera novog BM25 retrievala |
| Qdrant dense | provera da migracija nije promenila dense rezultat |
| Qdrant hybrid RRF | kandidat za novi aktivni backend |

Merimo postojeće metrike: Hit@1, Hit@5, Hit@10, MRR@10, Recall@10 i
cross-bank coverage. Dodatno beležimo ukupno vreme evaluacije i prosečno vreme
jednog lokalnog query-ja.

Trenutni hybrid rezultat koji treba dostići ili mu biti veoma blizu je:

```text
Hit@1:  11/28
Hit@5:  25/28
Hit@10: 26/28
MRR@10: 0,614
Recall@10: 0,855
```

## 8. Odluka posle evaluacije

Posle jednog završnog izveštaja biramo samo jednu od tri putanje:

1. Qdrant dense + sparse + RRF postaje aktivni retrieval backend ako zadrži
   približno isti ili bolji kvalitet i pojednostavi lokalni rad.
2. Qdrant koristimo za dense pretragu, a zadržavamo postojeći BM25S ako je novi
   sparse deo uzrok jasne regresije.
3. Privremeno zadržavamo postojeći retrieval ako Qdrant donosi ozbiljniju
   regresiju ili praktične lokalne probleme.

Mala razlika u jednoj metrici nije automatski neuspeh. Pregledaćemo koja su
konkretna pitanja izgubljena i da li je dobitak u brzini i jednostavnosti
vredniji od te razlike.

## 9. Šta dolazi tek posle toga

Ako Qdrant bude prihvaćen, sledeći logičan korak je stabilna retrieval funkcija
koju GPT-4o može da pozove kao alat. Zatim dolaze generisanje odgovora iz
originalnih tabela, citati i end-to-end evaluacija odgovora.

Reranker, query decomposition, LangChain i LangGraph uvodimo samo ako ta faza
pokaže potrebu. Time studentski projekat ostaje razumljiv, a svaki novi alat ima
jasnu svrhu.

## 10. Definition of done

Ova faza je završena kada:

- lokalna Qdrant baza može brzo da se otvori i koristi bez GPU-a;
- sadrži svih 5.565 zapisa i postojeće dense embeddinge;
- dense, sparse i hybrid pretraga rade iz postojećeg CLI-ja;
- table rezultat vraća celu originalnu tabelu;
- svih 30 pitanja je evaluirano jednom, na kraju;
- rezultat je upoređen sa postojećim baseline-om;
- dokumentovana je odluka koji retrieval backend koristimo dalje.

## Zvanične reference

- [Qdrant Python client i Local Mode](https://github.com/qdrant/qdrant-client)
- [Qdrant collections i named vectors](https://qdrant.tech/documentation/manage-data/collections/)
- [Qdrant hybrid queries i RRF](https://qdrant.tech/documentation/search/hybrid-queries/)
- [Qdrant full-text i BM25 search](https://qdrant.tech/documentation/search/text-search/full-text-search/)
