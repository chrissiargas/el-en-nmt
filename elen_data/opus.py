"""Read OPUS moses-format corpora straight from the object store.

Zips are read with HTTP range requests, so nothing has to be downloaded to
disk first. If ``<cache_dir>/<corpus>_<version>.zip`` exists it is used
instead of the network.
"""
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

OPUS_API = 'https://opus.nlpl.eu/opusapi?source={src}&target={tgt}&preprocessing=moses&version=latest'
OPUS_ZIP = 'https://object.pouta.csc.fi/OPUS-{corpus}/{version}/moses/{src}-{tgt}.txt.zip'


def list_corpora(src='el', tgt='en', cache_file=None):
    """Return {corpus_name: metadata} for the latest version of every corpus."""
    if cache_file and Path(cache_file).exists():
        data = json.loads(Path(cache_file).read_text())
    else:
        with urllib.request.urlopen(OPUS_API.format(src=src, tgt=tgt), timeout=120) as r:
            data = json.load(r)
        if cache_file:
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_file).write_text(json.dumps(data))
    return {c['corpus']: c for c in data['corpora']}


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with retries."""

    def __init__(self, url, retries=5):
        self.url, self.retries, self.pos = url, retries, 0
        with self._open(urllib.request.Request(url, method='HEAD')) as r:
            self.size = int(r.headers['Content-Length'])

    def _open(self, req):
        for attempt in range(self.retries):
            try:
                return urllib.request.urlopen(req, timeout=120)
            except OSError:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buf):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end}'})
                with self._open(req) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
        buf[:len(data)] = data
        self.pos += len(data)
        return len(data)


def _open_zip(corpus, version, src, tgt, cache_dir):
    local = Path(cache_dir) / f'{corpus}_{version}.zip' if cache_dir else None
    if local and local.exists():
        return zipfile.ZipFile(local)
    url = OPUS_ZIP.format(corpus=corpus, version=version, src=src, tgt=tgt)
    # 32 MB range requests: ~2x the throughput of 8 MB ones (TLS/request setup dominates small reads).
    return zipfile.ZipFile(io.BufferedReader(HttpFile(url), 1 << 25))


def _member_lines(corpus, version, src, tgt, cache_dir, suffix):
    # Every stream gets its own ZipFile (and HTTP reader) so the parallel
    # members don't fight over one file position.
    z = _open_zip(corpus, version, src, tgt, cache_dir)
    name = f'{corpus}.{src}-{tgt}.{suffix}'
    if name not in z.namelist():
        return None
    return io.TextIOWrapper(io.BufferedReader(z.open(name), 1 << 20), encoding='utf-8', errors='replace',
                            newline='\n')


def iter_pairs(corpus, version, src='el', tgt='en', cache_dir=None):
    """Yield (src_line, tgt_line, score) tuples; score is None unless the zip ships a .scores file."""
    src_f = _member_lines(corpus, version, src, tgt, cache_dir, src)
    tgt_f = _member_lines(corpus, version, src, tgt, cache_dir, tgt)
    if src_f is None or tgt_f is None:
        raise FileNotFoundError(f'{corpus} {version}: moses files for {src}-{tgt} not found')
    score_f = _member_lines(corpus, version, src, tgt, cache_dir, 'scores')
    scores = (float(s) for s in score_f) if score_f is not None else iter(lambda: None, 0)
    for s, t, sc in zip(src_f, tgt_f, scores):
        yield s.rstrip('\n'), t.rstrip('\n'), sc
