"""Download the el-en dev/test sets used for validation, evaluation and decontamination.

  FLORES-200 dev     (997 sents)  -> data/test/flores200.dev.{el,en}      validation
  FLORES-200 devtest (1012 sents) -> data/test/flores200.devtest.{el,en}  main test set
  NTREX-128          (1997 sents) -> data/test/ntrex128.{el,en}           second test set (news; English-original)
"""
import argparse
import io
import tarfile
import urllib.request
from pathlib import Path

FLORES = 'https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz'
NTREX = 'https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128/newstest2019-{}.txt'


def fetch(url):
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='data/test')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with tarfile.open(fileobj=io.BytesIO(fetch(FLORES)), mode='r:gz') as tar:
        for split in ('dev', 'devtest'):
            for code, lang in (('ell_Grek', 'el'), ('eng_Latn', 'en')):
                member = f'./flores200_dataset/{split}/{code}.{split}'
                (out / f'flores200.{split}.{lang}').write_bytes(tar.extractfile(member).read())

    (out / 'ntrex128.en').write_bytes(fetch(NTREX.format('src.eng')))
    (out / 'ntrex128.el').write_bytes(fetch(NTREX.format('ref.ell')))

    for f in sorted(out.iterdir()):
        print(f'{f}: {sum(1 for _ in f.open(encoding="utf-8"))} lines')


if __name__ == '__main__':
    main()
