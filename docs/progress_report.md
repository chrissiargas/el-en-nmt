# Greek↔English NMT: data preprocessing progress report

*21 October 2026*

## Summary

- **Task:** one Greek↔English MT model covering both directions, aiming for the best quality possible.
  Research use only. Training data comes from [OPUS el-en](https://opus.nlpl.eu/corpora-search/el&en).
- **Starting point:** OPUS lists 114 el-en corpora with **361M sentence pairs**. Much of this is double-counted
  (the same corpus released under several names), mislabelled, or too noisy to use as is, and it is far more
  than we can store or train on.
- **Done:**
  - picked 26 main corpora plus 69 small ELRC/ELRA sets;
  - streamed **177M pairs** straight from OPUS without downloading the archives;
  - normalised, filtered, deduplicated and decontaminated them, keeping **15.4M candidate pairs** (5 GB on disk).
- **Expected final training set:** **about 12–12.7M pairs**, once the LaBSE alignment check and the
  per-corpus caps are applied.
- **Next:**
  1. LaBSE scoring.
  2. Assemble the training set.
  3. Build a small "premium" subset scored by a quality-estimation model.
  4. Zero-shot baselines.
  5. Fine-tune NLLB-3.3B and Llama-Krikri-8B.

Code: `elen_data/` (filters, OPUS streaming), `scripts/` (pipeline), `configs/corpora.yaml` (the selection).
Detailed reasoning: `docs/data_selection.md`.

---

## 1. What OPUS el-en actually contains

| Group | Corpora | Listed pairs | Comment |
|---|---|---:|---|
| Web-mined | HPLT v3, CCMatrix, ParaCrawl v9, CCAligned, MaCoCu v2, WikiMatrix | ~139M | automatically aligned web text, quality varies a lot |
| Same corpus under another name | NLLB (= CCMatrix), MultiHPLT (= HPLT), MultiMaCoCu (= MaCoCu) | ~108M | identical files, counted twice |
| Subtitles | OpenSubtitles v2024, QED | ~77M | conversational, short lines, timing-based alignment |
| Institutional (EU) | DGT, EUbookshop, Europarl, JRC-Acquis, ECB, ELITR-ECA, EMEA, … | ~13M | clean but legal/administrative/medical and very repetitive |
| General curated | wikimedia, Wikipedia, TED2020, SETIMES, GlobalVoices, SciPar, Tatoeba, Books | ~3M | human translations across many domains; the most valuable part |
| Small ELRC/ELRA | ~75 sets (Greek/Cypriot government, health, press releases) | ~0.4M | about half are copies of each other |
| Word lists / software | XLEnt, WikiTitles, KDE4, GNOME, translatewiki, Ubuntu | ~5M | entity lists and UI strings, not sentences |

So about a third of the "361M" is double-counted, and most of the rest is web-mined or subtitles.

## 2. Which datasets we chose and why

The plan was a **diverse, high-quality set of about 10M pairs**, not "as much as possible":
- Disk, download time and GPU time are the limits, not data availability.
- For fine-tuning a pretrained model, clean and diverse data matters more than volume.

Corpora are processed **in order of trust** and deduplicated across all corpora. When the same pair appears in
several corpora, it is credited to the most trusted one.

### Tier 1: curated, mostly human-translated → keep everything

| Corpus | Domain | Why |
|---|---|---|
| Europarl | parliament speeches | large, clean, professionally translated |
| TED2020 | talks, general topics | spoken style, many topics |
| GlobalVoices, SETIMES | news | news domain, close to our NTREX test set |
| wikimedia, Wikipedia | encyclopedic | human (Content Translation) Wikipedia translations, close to FLORES |
| ELRC-5067-SciPar | thesis abstracts | scientific and technical vocabulary |
| ELITR-ECA, ECB, JRC-Acquis, EUconst, TildeMODEL, SPC, ECDC | EU institutions, finance, health | clean but small |
| Tatoeba, Books | everyday sentences, literature | short everyday sentences, literary style |
| 69 small ELRC/ELRA sets | Greek/Cypriot public administration | Greek-specific content (Greek institutions, Cyprus) |

### Tier 2: curated but huge or repetitive → capped

| Corpus | Listed | Cap | Why capped |
|---|---:|---:|---|
| DGT | 5.8M | 800K | EU legislation; formulaic, would take over the training data |
| ELRC-2711-EMEA | 0.78M | 300K | drug leaflets, extremely repetitive |
| EUbookshop | 4.0M | 400K | text extracted from PDFs, noisy |
| QED | 0.55M | 200K | educational subtitles |
| LinguaTools-WikiTitles | 1.85M | 100K | a small dose of named entities (Greek↔Latin transliteration) |

### Tier 3: web-mined and subtitles → strict filters, random sample

| Corpus | Listed | Cap | Why included | How it is sampled |
|---|---:|---:|---|---|
| MaCoCu v2 | 3.6M | 1.5M | cleanest mined corpus (already filtered by Bicleaner at release) | random |
| CCMatrix | 49M | 1.5M | sorted by LASER similarity score, so the head is the best part | top by score |
| ParaCrawl v9 | 21M | 1.5M | broad web domains | random |
| HPLT v3 | 56M | 1.0M | most recent crawl, new vocabulary | random |
| OpenSubtitles v2024 | 77M | 1.5M | the only large source of informal, dialogue-style language | random |

### Left out

| Corpus | Reason |
|---|---|
| NLLB, MultiHPLT, MultiMaCoCu | byte-identical to CCMatrix, HPLT, MaCoCu |
| NeuLab-TedTalks | same talks as TED2020 |
| ELRC-EMEA, ELRC_2682, EMEA | copies, or the same domain as ELRC-2711-EMEA |
| CCAligned | very poor alignment; audited as mostly noise (Kreutzer et al., 2022) |
| WikiMatrix | noisy; Wikipedia is already covered by human-translated wikimedia |
| XLEnt | automatically extracted entity pairs, not sentences |
| KDE4, GNOME, Ubuntu, translatewiki | software UI strings (placeholders, `&File`, `%s`) |
| bible-uedin | archaic language; religious text is already over-represented at the top of CCMatrix |

## 3. Problems in the data and how we handled them

The numbers come from our own build logs (`data/filtered/stats/*.json`).

| # | Problem | Evidence | What we did |
|---|---|---|---|
| 1 | **Duplicate corpora under different names** | NLLB and CCMatrix have identical file sizes; ELRA-W0xxx sets copy ELRC sets | Excluded the copies. Global deduplication catches the rest: e.g. ELRC-EUR_LEX kept **63 of 23,024** pairs and ELRA-W0243 kept **41 of 21,247**; **7 small sets turned out to be 100% duplicates** |
| 2 | **Wrong OPUS metadata** | ELRC-EMEA is listed as 13.7M pairs but is the same 0.78M file as ELRC-2711-EMEA; several ELRC sets have shifted columns | Checked the real file sizes and excluded the broken entries |
| 3 | **Domain imbalance** (EU legal/medical) | DGT alone is 5.8M pairs; 28% of DGT is internal duplicates | Per-corpus caps (Tier 2), so no single domain dominates |
| 4 | **Untranslated text** (source = target, or English on the Greek side) | Wikipedia: **21%** copies + 11% non-Greek; ELITR-ECA: 10% copies; ECB: 14% non-Greek | `copy` rule, plus a **script check**: the Greek side must be mostly Greek letters and the English side mostly Latin. Because Greek and English use different alphabets, this replaces a language-ID model |
| 5 | **Misaligned pairs in web-mined data** | ParaCrawl and HPLT: **12%** of pairs have different numbers on the two sides | Stricter "mined" filter profile. A **number-consistency check** (`3,5` = `3.5`, but `150` ≠ `180`). CCMatrix taken by alignment score. LaBSE similarity check (step 6 below) |
| 6 | **Misalignment even in "clean" corpora** | Europarl has whole sessions shifted by one sentence (~2.5% of pairs) | LaBSE similarity is applied to **all** corpora, not only mined ones |
| 7 | **Noisy text extracted from PDFs** | EUbookshop: 20% length-ratio failures + 18% number mismatches; only 45% pass | Strict profile, capped to 400K |
| 8 | **Subtitle problems** | OpenSubtitles: **21%** "one-to-many" (e.g. «Εντάξει.» paired with 40 different English lines); `<i>` tags, `{\an8}` | Remove markup. Allow at most 2 different translations per sentence. Shorter maximum length |
| 9 | **Encoding problems** | the Greek question mark U+037E vs `;`, mojibake (`Î±`, `â€™`), zero-width characters | Unicode NFC normalisation, removal of control characters, mojibake detection |
| 10 | **Web noise** | ALL-CAPS Greek without accents (menus, headings); repeated words ("Αρχική Αρχική Αρχική") | `all_caps` and `repetition` rules for mined data |
| 11 | **Topic bias at the top of CCMatrix** | the highest-scoring rows are mostly religious publications | Capped at 1.5M; bible-uedin excluded; to check after LaBSE scoring |
| 12 | **Test-set leakage** | test sentences also appear in the training corpora | Every pair whose Greek *or* English side matches a FLORES-200 or NTREX-128 sentence is removed. Matching ignores case and punctuation. **20K pairs removed** |
| 13 | **Storage and download limits** | HPLT alone is a 6.6 GB zip (~19 GB unzipped); we had ~20 GB free | Corpora are **streamed over HTTP range requests** and filtered on the fly. 177M pairs were processed without storing any archive. Random sampling reads only as much as each cap needs |

### Filtering pipeline (applied to every pair)

```
OPUS zip (streamed)
  → normalise        NFC, strip markup/control chars, squeeze whitespace
  → rule filters     3 strictness profiles: clean / mined / subs
                     (empty, too long, length ratio, copy, mojibake, script check,
                      too few letters, ALL CAPS, repetition, number mismatch)
  → decontaminate    drop anything matching FLORES-200 dev/devtest or NTREX-128
  → deduplicate      global across all corpora, ignoring case/punctuation;
                     ≤ 2 translations per sentence
  → sample           keep all (Tier 1) / random sample / top by score, up to the cap (+30% headroom)
  → LaBSE score      cosine similarity of multilingual sentence embeddings      ← next step
  → assemble         threshold + caps + shuffle → data/train.{el,en,corpus}     ← next step
```

The rules are unit-tested (`tests/test_textfilters.py`). Every rejection is counted per corpus and per reason.

## 4. Results so far

Over all corpora: 177.4M pairs read → **15.4M kept** (this includes 30% headroom for the LaBSE step).

Rejections by reason:

| Reason | Pairs |
|---|---:|
| one_to_many | 2.99M |
| duplicate | 2.80M |
| number_mismatch | 2.34M |
| length_ratio | 0.96M |
| el_not_greek | 0.94M |
| all_caps | 0.22M |
| low_alpha | 0.16M |
| copy | 0.14M |
| everything else | 0.17M |

How to read the per-corpus table:
- *Examined* is how many pairs were actually checked. For random-sampled corpora this is a random subset large
  enough to fill the cap.
- *Pass rate* is the share of examined pairs that survived the filters and deduplication.

| Corpus | Listed in OPUS | Examined | Pass rate | Kept now | Final target | Top rejection reasons |
|---|---:|---:|---:|---:|---:|---|
| Europarl | 1,292,180 | 1,292,180 | 96% | 1,243,560 | all | duplicate 1.6%, one_to_many 0.9% |
| TED2020 | 269,407 | 269,407 | 96% | 259,773 | all | duplicate 1.4%, empty 0.9% |
| GlobalVoices | 120,421 | 120,421 | 90% | 108,028 | all | empty 4.1%, el_not_greek 2.5% |
| SETIMES | 227,168 | 227,168 | 98% | 222,125 | all | duplicate 1.1%, el_not_greek 0.5% |
| wikimedia | 911,307 | 911,307 | 81% | 740,744 | all | el_not_greek 7.4%, duplicate 4.5% |
| Wikipedia | 104,076 | 104,076 | 66% | 68,363 | all | copy 20.6%, el_not_greek 11.3% |
| ELRC-5067-SciPar | 742,987 | 742,987 | 99% | 735,954 | all | one_to_many 0.4%, too_long 0.2% |
| ELITR-ECA | 381,561 | 381,561 | 80% | 305,035 | all | copy 10.3%, duplicate 4.3% |
| ECB | 102,986 | 102,986 | 59% | 61,171 | all | el_not_greek 13.8%, copy 10.8% |
| Tatoeba | 27,103 | 27,103 | 85% | 22,949 | all | one_to_many 14.4% |
| Books | 1,285 | 1,285 | 96% | 1,230 | all | length_ratio 4.1% |
| JRC-Acquis | 17,717 | 17,717 | 62% | 11,003 | all | copy 17.5%, duplicate 12.7% |
| EUconst | 9,990 | 9,990 | 53% | 5,282 | all | duplicate 18.9%, copy 18.8% |
| TildeMODEL | 5,238 | 5,238 | 81% | 4,244 | all | duplicate 15.1% |
| SPC | 8,181 | 8,181 | 63% | 5,118 | all | copy 15.8%, duplicate 15.1% |
| ECDC | 2,531 | 2,531 | 89% | 2,262 | all | duplicate 6.6% |
| DGT | 5,815,620 | 4,161,630 | 59% | 1,040,000 | 800,000 | duplicate 27.9%, one_to_many 6.1% |
| ELRC-2711-EMEA | 781,988 | 781,988 | 85% | 390,000 | 300,000 | one_to_many 6.0%, el_not_greek 4.0% |
| EUbookshop | 4,022,952 | 2,079,081 | 45% | 520,000 | 400,000 | length_ratio 19.9%, number_mismatch 18.0% |
| QED | 550,438 | 550,438 | 59% | 260,000 | 200,000 | duplicate 25.5%, one_to_many 6.0% |
| LinguaTools-WikiTitles | 1,850,804 | 520,402 | 28% | 130,000 | 100,000 | one_to_many 47.2%, length_ratio 11.8% |
| MaCoCu | 3,583,978 | 3,583,978 | 89% | 1,950,000 | 1,500,000 | number_mismatch 6.1%, el_not_greek 3.0% |
| CCMatrix | 49,262,631 | 2,385,907 | 82% | 1,950,000 | 1,500,000 | duplicate 10.5%, number_mismatch 2.6% |
| ParaCrawl | 21,402,471 | 7,802,072 | 78% | 1,950,000 | 1,500,000 | number_mismatch 12.2%, one_to_many 5.7% |
| HPLT | 55,649,558 | 5,199,226 | 74% | 1,300,000 | 1,000,000 | number_mismatch 12.4%, el_not_greek 7.7% |
| OpenSubtitles | 76,818,359 | 7,798,230 | 66% | 1,950,000 | 1,500,000 | one_to_many 20.8%, duplicate 6.8% |
| 69 small ELRC/ELRA sets | 386,566 | 352,579 | 43% | 151,649 | all | mostly duplicates of each other |

**Expected composition of the final set** (before the LaBSE removals in Tier 1):

| Tier | Pairs | Share |
|---|---:|---:|
| Tier 1, curated | ~3.8M | ~30% |
| Tier 2, capped curated | ~1.8M | ~14% |
| Tier 3, web-mined + subtitles | ~7.0M | ~55% |
| Small ELRC/ELRA sets | ~0.15M | ~1% |
| **Total** | **≈12.7M** | |

## 5. Open issues and limitations

1. **LaBSE not run yet.** The thresholds in the config (clean 0.6 / mined 0.7 / subtitles 0.65) are first
   guesses. They need to be calibrated on known-good pairs (Europarl/TED2020) and checked by reading the
   lowest-scoring pairs.
2. **Test-set decontamination is too aggressive for very short sentences.** NTREX contains lines like
   "Oh, no." and "Remember?". Removing every training pair that contains them dropped 19.5K OpenSubtitles pairs.
   This is not real contamination. Fix: only decontaminate test sentences of ≥ 5 words. (Small effect; not urgent.)
3. **The number-mismatch rule may be too strict.** It removed 2.3M pairs. Greek often writes numbers as words or
   ordinals (e.g. "3ος" vs "third"), so some of these are good pairs. We should read a sample of ~200 before
   relaxing the rule.
4. **Test sets are English-original.** FLORES-200 and NTREX-128 were both translated *from* English, so for
   el→en the Greek input is itself a translation. This should be stated in the results, and ideally we add a
   small Greek-original test set.
5. **Overlap with the pretrained models' own training data.** NLLB was trained on CCMatrix-style mined data, so
   fine-tuning NLLB on our mined tier will probably add little. The curated tiers are where we expect gains.
6. **Disk.** 9 GB free on the dev machine. The assembled training set (~4 GB) fits, but the raw filtered
   corpora should move to the A100 machine afterwards.

## 6. How to proceed

### Data: short term

1. **Calibrate and run LaBSE** on all 15.4M pairs (~2 h on the 2080 Ti) → `scripts/score_labse.py`.
2. **Assemble** the training set → `scripts/assemble.py` → `data/train.{el,en,corpus}`. The `.corpus` file
   records which corpus each line came from, for domain tags and ablations.
3. **Premium subset (~200K pairs)**: score Tier 1 and Tier 2 with a reference-free quality-estimation model
   (CometKiwi), then pick the best pairs with per-corpus quotas so the subset stays diverse. Used for LLM
   fine-tuning and for a final "polish" fine-tune of the NMT model.
4. **Manual check**: a native speaker reads ~100 random pairs per tier.
5. Fix open issue 2 and look into issue 3; move the data to the A100 machine.

### Modelling: on 4×A100

| Step | What | Why |
|---|---|---|
| 0. Zero-shot baselines | Evaluate OPUS-MT, NLLB-200 (1.3B/3.3B), MADLAD-400 (3B/7B), Llama-Krikri-8B-Instruct (few-shot) in both directions on FLORES devtest and NTREX | Choose the starting model and set the numbers to beat (1–2 days) |
| A. Dedicated MT model | Fine-tune NLLB-3.3B (or MADLAD-3B) on the ~12M set in both directions, then briefly on the premium set | Strong, well-understood baseline; also serves as the back-translation model |
| B. Translation LLM (main bet) | Fine-tune **Llama-Krikri-8B** (already adapted to Greek) on the premium set following the ALMA recipe, then preference tuning (CPO) | Recent work (ALMA, Tower) shows LLMs fine-tuned on small, very clean data match or beat large NMT models. Should help most with fluent Greek output (en→el) |
| C. Extras | Back-translation of Greek monolingual text; reranking/MBR decoding with a quality-estimation model; checkpoint averaging | Additional gains, if time allows |

**Evaluation:**
- Automatic metrics: chrF++, BLEU (sacreBLEU), COMET-22 and MetricX, in both directions on FLORES-200 devtest
  and NTREX-128.
- Plus a small human evaluation by native speakers.
- We don't select models with the same metric used for data selection or reranking, to avoid metric bias.

**Possible research contribution:** the pipeline already supports the ablations for a study of data selection
for Greek:
- rules only vs. +LaBSE vs. +quality estimation;
- data size (10% / 30% / 100%);
- curated-only vs. curated + web-mined.

## 7. Questions for the supervisor

1. Is "best quality, both directions, one model" the right goal, or is a smaller deployable model also expected?
2. Are we confirmed to have the 4×A100 (40 or 80 GB?), and for roughly how long?
3. Do we stay research-only? Some selected data (OpenSubtitles, TED2020) and models (NLLB) are non-commercial.
4. Is there a Greek-original test set (in-house or from a shared task) we could add?
5. Is a human evaluation feasible, e.g. 2 native speakers × 200 sentences?
