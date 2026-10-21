import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elen_data.textfilters import PROFILES, check_pair, dedup_key, normalize  # noqa: E402

CLEAN, MINED = PROFILES['clean'], PROFILES['mined']


def test_normalize():
    assert normalize('  Τι κάνεις;​ <i>τώρα</i>\t ') == 'Τι κάνεις; τώρα'
    assert normalize('{\\an8}Γεια') == 'Γεια'


def test_dedup_key_ignores_case_punct_space():
    assert dedup_key('Hello, world!') == dedup_key('hello world')
    assert dedup_key('Γεια σου.') != dedup_key('Γεια σας.')


@pytest.mark.parametrize('el,en', [
    ('Η Επιτροπή εγκρίνει την πρόταση.', 'The Commission approves the proposal.'),
    ('Εντάξει.', 'Okay.'),
    ('Το ποσοστό ήταν 3,5% το 2020.', 'The rate was 3.5% in 2020.'),
    ('Ο Οδυσσέας ζει στη Θεσσαλονίκη.', 'Odysseus lives in Thessaloniki.'),
])
def test_good_pairs_pass(el, en):
    assert check_pair(el, en, MINED) is None
    assert check_pair(el, en, CLEAN) is None


@pytest.mark.parametrize('el,en,reason', [
    ('', 'Hello', 'empty'),
    ('Η Επιτροπή εγκρίνει την πρόταση.', 'The Commission approves the proposal and also many other things today.',
     'length_ratio'),
    ('Microsoft Windows', 'Microsoft Windows', 'copy'),
    ('Î— Î•Ï€Î¹Ï„ÏÎ¿Ï€Î®', 'The Commission', 'mojibake'),
    ('The Commission approves.', 'The Commission approves it.', 'el_not_greek'),
    ('Η Επιτροπή εγκρίνει.', 'Η Επιτροπή εγκρίνει την.', 'en_not_latin'),
    ('Η ΕΠΙΤΡΟΠΗ ΕΓΚΡΙΝΕΙ ΤΗΝ ΠΡΟΤΑΣΗ', 'THE COMMISSION APPROVES THE PROPOSAL', 'all_caps'),
    ('Πλήρωσα 150 ευρώ.', 'I paid 180 euros.', 'number_mismatch'),
    ('Πίνακας 12345 67890 1234 55', 'Table 12345 67890 1234 55', 'low_alpha'),
    ('Αρχική Αρχική Αρχική Αρχική σελίδα', 'Home Home Home Home page', 'repetition'),
])
def test_bad_pairs_rejected(el, en, reason):
    assert check_pair(el, en, MINED) == reason


def test_clean_profile_is_lenient():
    assert check_pair('Η ΕΠΙΤΡΟΠΗ ΕΓΚΡΙΝΕΙ ΤΗΝ ΠΡΟΤΑΣΗ', 'THE COMMISSION APPROVES THE PROPOSAL', CLEAN) is None
    assert check_pair('Πλήρωσα 150 ευρώ.', 'I paid 180 euros.', CLEAN) is None
