# coding=utf-8

import re

from phonology import replace_umlauts

# The idea is to use a rule-based detection to curate a "gold list", and to
# then use machine learning to detect non-delimited compound boundaries, the
# performance of which will be evaluated against the rule-curated gold list.

# Ulimately:
#   Token.is_compound  --> hand-verified compound
#   Token.is_rule_based_compound  --> rule-based compound (is_test_compound)
#   Token.is_machine_learned_compound  --> machine-learned compound
#   Token.is_nondelimited_compound()  --> subset of compounds that do not
#                                         contain spaces of hyphens

# Compound detection:
#   * word length
#   * the presence of /uo/ and /yö/ sequences word-medially
#   * splitting /oy/ and /ay/ sequences
#   * contradictory front/back vowel harmony


# Rule-based ------------------------------------------------------------------

def detect(word):
    '''Detect if a word is a non-delimited compound.'''
    return bool(re.search(r'(-| |=)', word)) or bool(
        # any word that possesses both front and back vowels is a compound
        re.search(r'[AyO]+', word) and re.search(r'[auo]+', word))


def split(word):
    '''Insert syllable breaks at non-delimited compound boundaries.'''
    litmus = [
        # any syllable with a /uo/ or /yö/ nucleus denotes a word boundary,
        # always appearing word-initially
        (0, r'[ieAyOauo]+[^ -]*([^ieAyOauo]{1}(uo|yO))[^ieAyOauo]+'),

        # any word-medial sequence of /oy/ or /ay/ is an unnatural diphthong
        # and denotes a syllable boundary (this "bleeds" T4)
        (1, r'[ieAyOauo]+[^ieAyOauo]+[ieAyOauo]?(oy|ay)[ieAyOauo]?[^ieAyOauo]+[^$]'),  # noqa

        # any vowel sequence consisting of both front and back vowels denotes a
        # syllable boundary (cf. vowel harmony)
        (1, r'(Aa|Au|Ao|ya|yu|yo|Oa|Ou|Oo|aA|uA|oA|uy|aO|uO|oO)'),
        ]

    for i, pattern in litmus:
        offset = 0

        for vv in re.finditer(pattern, word):
            j = vv.start(1) + offset + i
            word = word[:j] + '=' + word[j:]
            offset += 1

    litmus = [
        # any word that possesses both front and back vowels is a compound
        r'[AyO]+([^ieAyOauo -=]+)[auo]+',
        r'[auo]+([^ieAyOauo -=]+)[AyO]+',

        # FROM GOOD TO BAD (3)
        # test 1                  rules 1    p / r          test 1                  rules 1       p / r      compound     # noqa
        # ----------------------  ---------  ---------  --  ----------------------  ------------  ---------  ----------   # noqa
        # ää.nes.tys.pro.sent.ti  T1af       1.0 / 1.0  >   ää.nes.tysp.ro.sent.ti  T1ac | T1ab   0.0 / 0.0  C            # noqa
        # käyt.tö.pro.sent.ti     T1abd      1.0 / 1.0  >   käyt.töp.ro.sent.ti     T1abc | T1ab  0.0 / 0.0  C            # noqa
        # ly.hyt.proo.saa         T1abf      1.0 / 1.0  >   ly.hytp.roo.saa         T1abc | T1ab  0.0 / 0.0  C            # noqa
        ]

    for pattern in litmus:
        offset = 0

        for vv in re.finditer(pattern, word):
            i = vv.end(1) - 1 + offset
            word = word[:i] + '=' + word[i:]
            offset += 1

    return word.replace('==', '=')

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    words = [
        # u'seurakuntayhtymä',
        # u'emoyhtiö',
        # u'lentoyhtiö',
        # u'tietoyhteiskunnan',
        # u'hääyöaie',
        # u'äänestysprosentti',
        u'york',
        ]

    for word in words:
        print replace_umlauts(split(replace_umlauts(word)), put_back=True)
