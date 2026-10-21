"""Normalisation and rule-based filters for Greek-English sentence pairs.

Greek and English use different scripts, so a per-side script check works
almost as well as a language-ID model here, and needs no model at all.
"""
import re
import unicodedata
from dataclasses import dataclass

import regex
import xxhash

_CTRL = regex.compile(r'[\p{Cc}\p{Cf}--[\t\n]]', regex.V1)  # control/format chars (incl. zero-width, BOM)
_SPACE = regex.compile(r'\s+')
_TAG = regex.compile(r'</?[A-Za-z][^<>]{0,40}>|\{\\[^{}]{0,20}\}')  # <i>, </font>, {\an8}
# Counting via sub() on the complement is several times faster than findall().
_NON_LETTER = regex.compile(r'\P{L}+')
_NON_GREEK = regex.compile(r'\P{Greek}+')
_NON_LATIN = regex.compile(r'\P{Latin}+')
_NON_UPPER = regex.compile(r'\P{Lu}+')
_REPEAT = re.compile(r'(\w)\1{5,}|\b(\w+)(?:\W+\2\b){3,}')  # "aaaaaa", "home home home home"
_MOJIBAKE = re.compile(r'�|â€|Ã[\x80-\xBF]|[ÎÏ][\x80-\xBF‘-›Œ-Ÿ]')
_NUMBER = regex.compile(r'\d+(?:[.,\s]\d+)*')
_KEY = regex.compile(r'[\p{P}\p{S}\s]+')


def normalize(text):
    """NFC (this also maps U+037E Greek question mark to ';'), drop markup/control chars, squeeze spaces."""
    text = unicodedata.normalize('NFC', text)
    text = _CTRL.sub('', text)
    text = _TAG.sub('', text)
    return _SPACE.sub(' ', text).strip()


def dedup_key(text):
    """64-bit hash of a text ignoring case, punctuation and whitespace."""
    return xxhash.xxh64_intdigest(_KEY.sub('', text.casefold()))


@dataclass
class FilterProfile:
    max_words: int = 150
    max_chars: int = 1000
    max_len_ratio: float = 2.5   # char-length ratio, checked once the longer side has >= 20 chars
    min_greek: float = 0.5       # share of letters in the el side that are Greek
    max_greek_in_en: float = 0.1
    min_latin_in_en: float = 0.5
    min_alpha: float = 0.25      # share of characters that are letters, per side
    max_upper: float = 1.0       # share of uppercase letters (web data has lots of ALL-CAPS Greek without accents)
    check_numbers: bool = False  # drop pairs whose digit sequences differ


PROFILES = {
    # Curated/institutional data: only remove things that are clearly broken.
    'clean': FilterProfile(),
    # Web-mined data: stricter.
    'mined': FilterProfile(max_words=120, max_len_ratio=2.0, min_greek=0.7, max_greek_in_en=0.05,
                           min_latin_in_en=0.7, min_alpha=0.4, max_upper=0.7, check_numbers=True),
    # Subtitles: short lines are fine, but alignment drift is common.
    'subs': FilterProfile(max_words=60, max_len_ratio=2.5, min_greek=0.7, max_greek_in_en=0.05,
                          min_latin_in_en=0.7, min_alpha=0.3, max_upper=0.7, check_numbers=True),
}


def _numbers(text):
    return sorted(regex.sub(r'\D', '', n) for n in _NUMBER.findall(text))


def check_pair(el, en, p):
    """Return None if the (already normalised) pair passes, else a short rejection reason."""
    if not el or not en:
        return 'empty'
    if len(el) > p.max_chars or len(en) > p.max_chars or \
            el.count(' ') + 1 > p.max_words or en.count(' ') + 1 > p.max_words:
        return 'too_long'
    longer, shorter = max(len(el), len(en)), min(len(el), len(en))
    if longer >= 20 and longer > p.max_len_ratio * shorter:
        return 'length_ratio'
    if el.casefold() == en.casefold():
        return 'copy'
    if _MOJIBAKE.search(el) or _MOJIBAKE.search(en):
        return 'mojibake'

    el_l, en_l = _NON_LETTER.sub('', el), _NON_LETTER.sub('', en)
    if not el_l or not en_l:
        return 'no_letters'
    if len(el_l) < p.min_alpha * len(el) or len(en_l) < p.min_alpha * len(en):
        return 'low_alpha'
    if len(_NON_GREEK.sub('', el_l)) < p.min_greek * len(el_l):
        return 'el_not_greek'
    if len(_NON_GREEK.sub('', en_l)) > p.max_greek_in_en * len(en_l) or \
            len(_NON_LATIN.sub('', en_l)) < p.min_latin_in_en * len(en_l):
        return 'en_not_latin'
    if p.max_upper < 1.0 and (
            (len(el_l) > 10 and len(_NON_UPPER.sub('', el_l)) > p.max_upper * len(el_l)) or
            (len(en_l) > 10 and len(_NON_UPPER.sub('', en_l)) > p.max_upper * len(en_l))):
        return 'all_caps'
    if _REPEAT.search(el) or _REPEAT.search(en):
        return 'repetition'
    if p.check_numbers and _numbers(el) != _numbers(en):
        return 'number_mismatch'
    return None
