# Greek–English data selection (OPUS)

Snapshot of [OPUS el-en](https://opus.nlpl.eu/corpora-search/el&en), October 2026: **114 corpora,
361M sentence pairs** (as listed). The real amount is much smaller, and most of it is not usable as is.

## 1. What is actually in OPUS el-en

| Group | Corpora | Listed pairs | Notes |
|---|---|---:|---|
| Web-mined | HPLT v3, CCMatrix, ParaCrawl v9, CCAligned, MaCoCu v2, WikiMatrix | ~139M | automatic alignment, quality varies a lot |
| Exact aliases | NLLB (= CCMatrix), MultiHPLT (= HPLT), MultiMaCoCu (= MaCoCu) | ~108M | same files, counted twice |
| Subtitles | OpenSubtitles v2024, QED | ~77M | conversational, short lines, alignment drift |
| Institutional | DGT, EUbookshop, Europarl, JRC-Acquis, ECB, ELITR-ECA, EUconst, EMEA, ... | ~13M | clean, but legal/EU/medical and very repetitive |
| General curated | wikimedia, Wikipedia, TED2020, SETIMES, GlobalVoices, SciPar, Tatoeba, Books | ~3M | the most valuable part |
| Small ELRC/ELRA | ~75 sets (Greek/Cypriot government, health, press releases) | ~0.4M | about half are mirrors of each other |
| Lexical / software | XLEnt, WikiTitles, KDE4, GNOME, translatewiki, Ubuntu, tldr-pages | ~5M | entity lists and UI strings, not sentences |

Things to know when reading the OPUS listing:

- **CCMatrix and NLLB are byte-identical** (same file sizes; NLLB is the CCMatrix re-release). Same for
  HPLT/MultiHPLT and MaCoCu/MultiMaCoCu.
- **The OPUS metadata is wrong for some ELRC sets.** `ELRC-EMEA` is listed with 13.7M pairs but is the
  same 0.78M-pair file as `ELRC-2711-EMEA` and `ELRC_2682`. `ELRC-presscorner_covid`,
  `ELRC-EUROPARL_covid`, `ELRC-antibiotic` and `ELRC-Press_Releases` have the same column shift. (The
  pipeline confirmed this: `ELRC-EUROPARL_covid` has 692 lines, and `ELRC-3290-EUROPARL_covid` is
  100 % duplicate of it.)
- **ELRA-W0xxx sets mirror ELRC-xxx sets** (e.g. `ELRA-W0205` = `ELRC-652-Greek_law`: 1,934/1,979 pairs duplicate).
- **CCMatrix is sorted by LASER margin score** and ships the scores. Every pair has margin ≥ 1.06
  (the CCMatrix paper's threshold); the top 1.5M have margin ≥ ~1.135, the top 10M ≥ 1.098. The
  highest-scoring rows are dominated by religious text (Jehovah's Witnesses publications), so the head is
  not domain-neutral.
- No other mined corpus ships scores in its OPUS moses zip (ParaCrawl/MaCoCu/WikiMatrix only have XCES
  alignment XML without scores), so those get scored with LaBSE here.

## 2. Constraints

One RTX 2080 Ti (11 GB, fp16 but no bf16), 31 GB RAM, ~21 GB free disk. A Transformer-base trained from
scratch on this GPU gets through roughly 2–3B target tokens in two days, i.e. 5–10 epochs over
~10M pairs. So the target is **~10M high-quality, diverse pairs**, not "as much as possible". All caps
scale with one flag (`--scale`) if the budget changes.

The disk constraint is why corpora are streamed from OPUS with HTTP range requests and filtered on the fly:
HPLT alone is a 6.6 GB zip (~19 GB unzipped).

## 3. Selection

Corpora are processed in this order and deduplicated globally, so a pair found in several corpora is
attributed to the most trusted one.

**Tier 1: curated, keep everything** (~4.4M listed)
Europarl, TED2020, GlobalVoices, SETIMES, wikimedia, Wikipedia, ELRC-5067-SciPar, ELITR-ECA, ECB,
Tatoeba, Books, JRC-Acquis, EUconst, TildeMODEL, SPC, ECDC, plus all small ELRC/ELRA sets (dedup removes
the mirrors).

**Tier 2: curated but repetitive domains, capped** so EU legal and medical text doesn't dominate
| Corpus | Listed | Cap | Why capped |
|---|---:|---:|---|
| DGT | 5.8M | 800K | EU legislation; huge and formulaic |
| ELRC-2711-EMEA | 0.78M | 300K | drug leaflets; extremely repetitive |
| EUbookshop | 4.0M | 400K | PDF-extracted, noisy; filtered with the strict profile |
| QED | 0.55M | 200K | educational subtitles; noisy, LaBSE-checked |
| LinguaTools-WikiTitles | 1.85M | 100K | small dose of named entities (Greek↔Latin transliteration) |

**Tier 3: web-mined and subtitles, strict rules + LaBSE**
| Corpus | Listed | Cap | Sampling |
|---|---:|---:|---|
| MaCoCu v2 | 3.6M | 1.5M | random (already bicleaner-filtered at release, cleanest mined set) |
| CCMatrix v1 | 49M | 1.5M | top by LASER margin |
| ParaCrawl v9 | 21M | 1.5M | random |
| HPLT v3 | 56M | 1.0M | random |
| OpenSubtitles v2024 | 77M | 1.5M | random; the only large source of informal/dialogue register |

For LaBSE-scored corpora the build keeps 1.3 × cap so that the similarity threshold has room to remove
pairs before the cap is applied.

**Left out**
| Corpus | Reason |
|---|---|
| NLLB, MultiHPLT, MultiMaCoCu | duplicates of CCMatrix, HPLT, MaCoCu |
| NeuLab-TedTalks | same talks as TED2020 |
| ELRC-EMEA, ELRC_2682, EMEA | duplicates / same domain as ELRC-2711-EMEA |
| CCAligned | document-level crawl with poor sentence alignment; audited as very noisy (Kreutzer et al., 2022, *Quality at a Glance*) |
| WikiMatrix | noisy; Wikipedia is covered by wikimedia (human Content Translation) |
| XLEnt | automatically mined entity pairs, not sentences |
| KDE4, GNOME, Ubuntu, translatewiki, tldr-pages | UI strings with placeholders and accelerator keys |
| bible-uedin | archaic register; religious text already over-represented in CCMatrix |

## 4. Filtering

Applied to every pair (`elen_data/textfilters.py`, `scripts/build_corpus.py`):

1. **Normalise**: Unicode NFC (also turns the Greek question mark U+037E into `;`), strip control and
   zero-width characters, subtitle/HTML markup (`<i>`, `{\an8}`), squeeze whitespace.
2. **Rules** (three strictness profiles: `clean`, `mined`, `subs`):
   - empty side, too long (>150 words / 1000 chars; 120 words for mined, 60 for subtitles);
   - character-length ratio > 2.5 (2.0 for mined);
   - source = target (untranslated copies);
   - mojibake (`Î±`, `â€™`, U+FFFD);
   - **script check**: the Greek side must be ≥ 50 % (mined: 70 %) Greek letters, the English side ≤ 10 %
     (mined: 5 %) Greek and ≥ 50 % (mined: 70 %) Latin. Because the two languages use different scripts,
     this replaces a language-ID model;
   - too few letters (tables, codes, number lists);
   - mined/subs only: ALL-CAPS text (common on Greek web pages, usually without accents) and **number
     mismatch** (the digits on both sides must match, so `3,5` = `3.5` but `150` ≠ `180`);
   - repeated characters/words (`Αρχική Αρχική Αρχική Αρχική`).
3. **Decontamination**: any pair whose Greek or English side matches a FLORES-200 dev/devtest or NTREX-128
   sentence (case- and punctuation-insensitive) is removed.
4. **Deduplication**, global across all corpora, case- and punctuation-insensitive:
   - exact duplicate pairs are removed;
   - at most 2 different translations per sentence on either side (stops "Εντάξει." → 40 variants of "Okay").
5. **Sampling**: uniform random sample (reservoir) or head for score-sorted CCMatrix, down to the cap.
6. **LaBSE similarity** (mined corpora, QED, OpenSubtitles): cosine of LaBSE embeddings; pairs below the
   threshold are dropped. The threshold is calibrated on clean data (Europarl/TED2020), see §5.

## 5. Results

_Filled in from `data/filtered/stats/*.json` and `data/train.stats.json` after the build._

## 6. Evaluation data

| Set | Size | Use |
|---|---:|---|
| FLORES-200 dev | 997 | validation / checkpoint selection |
| FLORES-200 devtest | 1012 | main test set (Wikipedia-style, multi-domain) |
| NTREX-128 | 1997 | second test set (news; English-original, so el→en has translationese input) |

All three are removed from the training data. Note that public OPUS-MT models were trained on OPUS
including Tatoeba, so do not compare against them on the Tatoeba test set.
