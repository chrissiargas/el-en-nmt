"""Combine the filtered corpora into one shuffled training set.

Pairs whose LaBSE similarity is below the threshold for their profile
(`min_labse` in the config) are dropped, then each corpus is randomly
downsampled to its cap.

Writes <train>.el, <train>.en, <train>.corpus (origin of every line, handy for
domain tags or ablations) and <train>.stats.json.

  python scripts/assemble.py
  python scripts/assemble.py --scale 0.3 --train data/train_small   # smaller training set from the same build
  python scripts/assemble.py --no-labse                             # rules + dedup only (for an ablation)
"""
import argparse
import json
import random
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/corpora.yaml')
    ap.add_argument('--filtered', default='data/filtered')
    ap.add_argument('--train', default='data/train', help='output prefix')
    ap.add_argument('--no-labse', action='store_true', help='ignore LaBSE scores')
    ap.add_argument('--scale', type=float, default=1.0, help='multiply every cap')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    entries = {c['name']: c for c in config['corpora']}
    small_profile = config.get('small_sets', {}).get('profile', 'clean')
    corpus_dir = Path(args.filtered) / 'corpora'
    # Configured corpora first (in config order), then the small sets.
    names = sorted((p.stem for p in corpus_dir.glob('*.el')),
                   key=lambda n: (list(entries).index(n) if n in entries else len(entries), n))

    rows, table = [], []
    for name in names:
        entry = entries.get(name, {})
        el = (corpus_dir / f'{name}.el').read_text(encoding='utf-8').splitlines()
        en = (corpus_dir / f'{name}.en').read_text(encoding='utf-8').splitlines()
        assert len(el) == len(en), name
        idx = list(range(len(el)))
        n_built = len(idx)

        if not args.no_labse:
            labse_path = corpus_dir / f'{name}.labse'
            if not labse_path.exists():
                raise SystemExit(f'{name} has no LaBSE scores: run scripts/score_labse.py first (or use --no-labse)')
            scores = [float(s) for s in labse_path.read_text().split()]
            assert len(scores) == len(el), name
            threshold = entry.get('min_labse', config['min_labse'][entry.get('profile', small_profile)])
            idx = [i for i in idx if scores[i] >= threshold]
        n_semantic = len(idx)

        cap = entry.get('cap')
        if cap is not None and len(idx) > cap * args.scale:
            idx = sorted(random.Random(f'{args.seed}-{name}').sample(idx, int(cap * args.scale)))
        rows.extend((el[i], en[i], name) for i in idx)
        table.append({'corpus': name, 'built': n_built, 'after_labse': n_semantic, 'final': len(idx)})

    random.Random(args.seed).shuffle(rows)
    prefix = Path(args.train)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for k, ext in enumerate(('el', 'en', 'corpus')):
        with open(f'{prefix}.{ext}', 'w', encoding='utf-8') as f:
            f.writelines(r[k] + '\n' for r in rows)

    total = len(rows)
    Path(f'{prefix}.stats.json').write_text(json.dumps({'total': total, 'corpora': table}, indent=1))
    print(f'{"corpus":32}{"built":>11}{"after LaBSE":>13}{"final":>11}{"share":>8}')
    small = [r for r in table if r['corpus'] not in entries]
    for r in [r for r in table if r['corpus'] in entries]:
        print(f'{r["corpus"]:32}{r["built"]:>11,}{r["after_labse"]:>13,}{r["final"]:>11,}{r["final"] / total:>8.1%}')
    s = sum(r['final'] for r in small)
    print(f'{f"{len(small)} small ELRC/ELRA sets":32}{sum(r["built"] for r in small):>11,}{"":>13}{s:>11,}{s / total:>8.1%}')
    print(f'{"TOTAL":32}{"":>11}{"":>13}{total:>11,}')
    print(f'wrote {prefix}.el / .en / .corpus')


if __name__ == '__main__':
    main()
