"""Score sentence pairs with LaBSE cosine similarity (catches misalignments the rules miss).

Writes <out>/corpora/<name>.labse (one float per line) for every built corpus,
or for the corpora given with --corpora. Prints score percentiles, so you can
calibrate the thresholds (`min_labse` in the config):

  python scripts/score_labse.py --corpora Europarl TED2020 --limit 50000   # what do good pairs score?
  python scripts/score_labse.py                                            # everything (~2k pairs/s on a 2080 Ti)
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def read_lines(path, limit):
    with open(path, encoding='utf-8') as f:
        lines = [line.rstrip('\n') for _, line in zip(range(limit or 10 ** 12), f)]
    return lines


def encode(model, sents, batch_size):
    # Sorting by length minimises padding; that alone roughly doubles throughput.
    order = np.argsort([len(s) for s in sents])
    emb = model.encode([sents[i] for i in order], batch_size=batch_size, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=len(sents) > 100_000)
    out = np.empty_like(emb)
    out[order] = emb
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='data/filtered')
    ap.add_argument('--corpora', nargs='+', help='default: every corpus in <out>/corpora')
    ap.add_argument('--limit', type=int, help='score only the first N pairs (for calibration)')
    ap.add_argument('--model', default='sentence-transformers/LaBSE')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--chunk', type=int, default=500_000, help='pairs encoded per chunk (bounds RAM use)')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    corpus_dir = Path(args.out) / 'corpora'
    names = args.corpora or sorted(p.stem for p in corpus_dir.glob('*.el'))
    model = SentenceTransformer(args.model, device='cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model.half()

    for name in names:
        out_path = corpus_dir / f'{name}.labse'
        if out_path.exists() and not args.force and not args.limit:
            print(f'{name}: already scored, skipping')
            continue
        el = read_lines(corpus_dir / f'{name}.el', args.limit)
        en = read_lines(corpus_dir / f'{name}.en', args.limit)
        scores = []
        for i in range(0, len(el), args.chunk):
            a = encode(model, el[i:i + args.chunk], args.batch_size)
            b = encode(model, en[i:i + args.chunk], args.batch_size)
            scores.append((a * b).sum(axis=1))
        scores = np.concatenate(scores) if scores else np.zeros(0)
        if not args.limit:
            tmp = out_path.with_suffix('.labse.tmp')
            np.savetxt(tmp, scores, fmt='%.4f')
            os.replace(tmp, out_path)
        pct = np.percentile(scores, [1, 5, 10, 25, 50]) if len(scores) else []
        share = {t: float((scores < t).mean()) for t in (0.6, 0.7, 0.75, 0.8)} if len(scores) else {}
        print(f'{name} ({len(scores):,} pairs)  percentiles 1/5/10/25/50: ' + ' '.join(f'{p:.3f}' for p in pct) +
              '  | share below ' + ', '.join(f'{t}: {s:.1%}' for t, s in share.items()), flush=True)


if __name__ == '__main__':
    main()
