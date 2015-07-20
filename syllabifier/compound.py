# coding=utf-8

import re

from phonology import replace_umlauts


def detect(word):
    # detect if the word is a non-delimited compound
    word = replace_umlauts(word)
    pattern = r'[ieAyOauo]+[^ -]*([^ieAyOauo]{1}(uo|yO))'

    return bool(re.search(pattern, word))


def split(word):
    # any syllable with a /uo/ or /yö/ nucleus denotes a word boundary, always
    # appearing word-initially
    pattern = r'[ieAyOauo]+[^ -]*([^ieAyOauo]{1}(uo|yO))'

    offset = 0

    for vv in re.finditer(pattern, word):
        i = vv.start(1) + offset
        word = word[:i] + '.' + word[i:]
        offset += 1

    return word
