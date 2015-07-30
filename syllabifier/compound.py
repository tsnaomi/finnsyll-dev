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

# Consider:
#   - /uo/ or /yö/ sequences
#   - word length
#   - where vowel harmony deviation appears in a word


# Rule-based ------------------------------------------------------------------

def detect(word):
    '''Detect if a word is a non-delimited compound.'''
    if '-' in word or ' ' in word:
        return True

    word = replace_umlauts(word)

    # any syllable with a /uo/ or /yö/ nucleus denotes a word boundary, always
    # appearing word-initially
    pattern = r'[ieAyOauo]+[^ -]*[^ieAyOauo]{1}(uo|yO)[^ieAyOauo]+'

    return bool(re.search(pattern, word))


def split(word):
    '''Insert syllable breaks at non-delimited compound boundaries.'''
    # any syllable with a /uo/ or /yö/ nucleus denotes a word boundary, always
    # appearing word-initially
    pattern = r'[ieAyOauo]+[^ -]*([^ieAyOauo]{1}(uo|yO))[^ieAyOauo]+'

    offset = 0

    for vv in re.finditer(pattern, word):
        i = vv.start(1) + offset
        word = word[:i] + '.' + word[i:]
        offset += 1

    return word

# -----------------------------------------------------------------------------
