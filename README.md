# Greek–English NMT

## Data

Training data comes from [OPUS el-en](https://opus.nlpl.eu/corpora-search/el&en). The selection and the
reasons behind it are in [docs/data_selection.md](docs/data_selection.md); the selection itself is
[configs/corpora.yaml](configs/corpora.yaml).

```bash
pip install -r requirements.txt

python scripts/fetch_testsets.py     # FLORES-200 dev/devtest + NTREX-128 -> data/test/
python scripts/build_corpus.py       # stream OPUS, filter, dedup, decontaminate, sample -> data/filtered/
python scripts/score_labse.py        # LaBSE similarity for every pair (GPU, ~2k pairs/s on a 2080 Ti)
python scripts/assemble.py           # threshold + caps + shuffle -> data/train.{el,en,corpus}
```

- Corpora are streamed straight from the OPUS object store with HTTP range requests, so the
  multi-GB zips never touch the disk. Put `<corpus>_<version>.zip` in `data/raw/` to read a local copy instead.
- `build_corpus.py --scale 0.1` gives a 10 % pilot set; `assemble.py --scale` shrinks the final set
  without rebuilding.
- Per-corpus filtering statistics (how many pairs each rule removed) are in `data/filtered/stats/`.
- Everything in `data/test/` is removed from the training data.

```bash
python -m pytest tests
```
