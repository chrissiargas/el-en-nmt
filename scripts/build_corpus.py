"""Stream the selected OPUS corpora, filter, deduplicate, decontaminate and sample them.

For every corpus in configs/corpora.yaml (in order) this writes
  <out>/corpora/<name>.el / .en   (+ .score when OPUS ships alignment scores)
  <out>/stats/<name>.json         counts per filtering stage / rejection reason

Corpora that already have output are not rebuilt (use --force); their pairs
are reloaded so deduplication against later corpora stays correct.

Example:
  python scripts/build_corpus.py                      # full selection
  python scripts/build_corpus.py --scale 0.1          # 10% of every cap, for a quick pilot
  python scripts/build_corpus.py --only TED2020 Tatoeba
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import xxhash
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elen_data.opus import iter_pairs, list_corpora  # noqa: E402
from elen_data.textfilters import PROFILES, check_pair, dedup_key, normalize  # noqa: E402


def load_selection(config, api):
    """Return the ordered list of corpus entries, with small_sets expanded from the OPUS listing."""
    entries = [dict(c) for c in config['corpora']]
    small = config.get('small_sets')
    if small:
        named = {c['name'] for c in entries} | set(small.get('exclude', []))
        for name, meta in sorted(api.items(), key=lambda kv: -int(kv[1]['alignment_pairs'] or 0)):
            if name.startswith(tuple(small['prefixes'])) and name not in named \
                    and 0 < int(meta['alignment_pairs'] or 0) <= small['max_pairs']:
                entries.append({'name': name, 'profile': small['profile'], 'take': 'all'})
    for c in entries:
        if c['name'] not in api:
            raise SystemExit(f'{c["name"]} is not an el-en corpus in OPUS')
        c.setdefault('version', api[c['name']]['version'])
    return entries


def pair_key(ke, kn):
    return xxhash.xxh64_intdigest(ke.to_bytes(8, 'little') + kn.to_bytes(8, 'little'))


class DedupState:
    """Global exact-duplicate index plus a cap on how many translations one sentence may have."""

    def __init__(self, contaminated, max_per_side):
        self.contaminated = contaminated
        self.max_per_side = max_per_side
        self.pairs = set()
        self.el = Counter()
        self.en = Counter()

    def check(self, ke, kn, kp):
        if ke in self.contaminated or kn in self.contaminated:
            return 'test_overlap'
        if kp in self.pairs:
            return 'duplicate'
        if self.el[ke] >= self.max_per_side or self.en[kn] >= self.max_per_side:
            return 'one_to_many'
        return None

    def add(self, ke, kn, kp):
        self.pairs.add(kp)
        self.el[ke] += 1
        self.en[kn] += 1

    def remove(self, ke, kn, kp):
        self.pairs.discard(kp)
        self.el[ke] -= 1
        self.en[kn] -= 1


def write_corpus(out_dir, name, pairs, has_score):
    """Write pairs atomically (tmp files + rename) so an interrupted run never leaves half a corpus."""
    paths = {ext: out_dir / f'{name}.{ext}' for ext in (['el', 'en', 'score'] if has_score else ['el', 'en'])}
    tmp = {ext: p.with_suffix(p.suffix + '.tmp') for ext, p in paths.items()}
    files = {ext: open(p, 'w', encoding='utf-8') for ext, p in tmp.items()}
    try:
        for el, en, score in pairs:
            files['el'].write(el + '\n')
            files['en'].write(en + '\n')
            if has_score:
                files['score'].write(f'{score}\n')
    finally:
        for f in files.values():
            f.close()
    for ext in paths:
        os.replace(tmp[ext], paths[ext])


def build_one(entry, cap, state, args, rng, raw_pairs):
    """Stream one corpus; returns (kept pairs, stats)."""
    profile = PROFILES[entry['profile']]
    take = entry.get('take', 'all')
    # For random sampling, only look at a Bernoulli subsample big enough to fill the cap after
    # filtering (`prefetch` x headroom), then reservoir-sample down to exactly `cap`.
    p_look = 1.0 if take != 'random' else min(1.0, args.prefetch * cap / max(raw_pairs, 1))
    reasons = Counter()
    n_read = n_looked = n_survived = 0
    kept = []  # (el, en, score, ke, kn, kp)

    stream = iter_pairs(entry['name'], entry['version'], cache_dir=args.cache_dir)
    for el, en, score in tqdm(stream, total=raw_pairs, desc=entry['name'], unit=' pairs', mininterval=5,
                              disable=raw_pairs < 100_000):
        n_read += 1
        if p_look < 1.0 and rng.random() >= p_look:
            continue
        n_looked += 1
        el, en = normalize(el), normalize(en)
        reason = check_pair(el, en, profile)
        if reason is None:
            ke, kn = dedup_key(el), dedup_key(en)
            kp = pair_key(ke, kn)
            reason = state.check(ke, kn, kp)
        if reason:
            reasons[reason] += 1
            continue

        n_survived += 1
        item = (el, en, score, ke, kn, kp)
        if take == 'random' and len(kept) >= cap:
            j = rng.randrange(n_survived)
            if j >= cap:
                continue
            state.remove(*kept[j][3:])
            kept[j] = item
        else:
            kept.append(item)
        state.add(ke, kn, kp)
        if take == 'head' and len(kept) >= cap:
            break

    if take == 'random':
        reasons['sampled_out'] = n_survived - len(kept)
    stats = {'corpus': entry['name'], 'version': entry['version'], 'profile': entry['profile'], 'take': take,
             'cap': cap, 'opus_pairs': raw_pairs, 'read': n_read, 'looked_at': n_looked,
             'kept': len(kept), 'rejected': dict(reasons.most_common())}
    return kept, stats


def reload_existing(corpus_dir, name, state):
    with open(corpus_dir / f'{name}.el', encoding='utf-8') as fe, open(corpus_dir / f'{name}.en', encoding='utf-8') as fn:
        for el, en in zip(fe, fn):
            ke, kn = dedup_key(el.rstrip('\n')), dedup_key(en.rstrip('\n'))
            state.add(ke, kn, pair_key(ke, kn))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/corpora.yaml')
    ap.add_argument('--out', default='data/filtered')
    ap.add_argument('--testsets', default='data/test', help='every file here is excluded from training data')
    ap.add_argument('--cache-dir', default='data/raw', help='use <corpus>_<version>.zip from here instead of streaming')
    ap.add_argument('--scale', type=float, default=1.0, help='multiply every cap (e.g. 0.1 for a pilot run)')
    ap.add_argument('--max-per-side', type=int, default=2,
                    help='max different translations kept for one (normalised) sentence on either side')
    ap.add_argument('--prefetch', type=float, default=4.0,
                    help='random sampling: look at prefetch*cap pairs, expecting filters to drop some')
    ap.add_argument('--only', nargs='+', help='process only these corpora (others are still reloaded for dedup)')
    ap.add_argument('--force', action='store_true', help='rebuild corpora that already have output')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out)
    corpus_dir, stats_dir = out / 'corpora', out / 'stats'
    corpus_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(open(args.config))
    api = list_corpora(cache_file=out / 'opus_el-en.json')
    entries = load_selection(config, api)

    contaminated = set()
    for f in sorted(Path(args.testsets).glob('*')):
        contaminated.update(dedup_key(normalize(line)) for line in open(f, encoding='utf-8'))
    contaminated.discard(dedup_key(''))
    state = DedupState(contaminated, args.max_per_side)
    print(f'{len(entries)} corpora selected; {len(contaminated)} test-set sentences excluded', flush=True)

    for i, entry in enumerate(entries):
        name = entry['name']
        done = (corpus_dir / f'{name}.el').exists() and (stats_dir / f'{name}.json').exists()
        if (args.only and name not in args.only) or (done and not args.force):
            if done:
                reload_existing(corpus_dir, name, state)
            continue
        # Each corpus gets its own RNG so --only / reruns sample the same pairs.
        rng = random.Random(f'{args.seed}-{name}')
        cap = entry.get('cap')
        if cap is not None:
            cap = int(cap * args.scale * config.get('oversample', 1.0))
        t0 = time.time()
        kept, stats = build_one(entry, cap, state, args, rng, int(api[name]['alignment_pairs'] or 0))
        stats['seconds'] = round(time.time() - t0)
        write_corpus(corpus_dir, name, ((el, en, sc) for el, en, sc, *_ in kept), kept and kept[0][2] is not None)
        (stats_dir / f'{name}.json').write_text(json.dumps(stats, indent=1))
        rej = ', '.join(f'{k} {v:,}' for k, v in stats['rejected'].items())
        print(f'[{i + 1}/{len(entries)}] {name}: kept {stats["kept"]:,} of {stats["looked_at"]:,} examined '
              f'({stats["read"]:,} read, {stats["seconds"]}s). Rejected: {rej or "none"}', flush=True)


if __name__ == '__main__':
    main()
