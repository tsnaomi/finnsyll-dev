# coding=utf-8

import re


def splitter(word):
    # any syllable with a /uo/ and /yO/ nuclei denotes a word boundary, always
    # appearing word-initially
    pattern = r'(?=[ieAyOauo]+[^ -]*([^ieAyOauo]{1}(uo|yO)))'

    offset = 0

    for vv in re.finditer(pattern, word):
        i = vv.start(1) + offset
        word = word[:i] + '.' + word[i:]
        offset += 1

    return word
