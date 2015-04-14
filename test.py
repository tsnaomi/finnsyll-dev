# coding=utf-8

from timeit import timeit
from syllabifier.falk.finnish_syllables import make_syllables
from syllabifier.v2 import syllabify


def timeit_test(words, f=False):
    if f:
        for word in words:
            make_syllables(word)

    else:
        for word in words:
            syllabify(word)


if __name__ == '__main__':
    WORDS = [
        u'kala',
        u'järjestäminenkö',
        u'kärkkyä',
        u'värväytyä',
        u'pamaushan',
        u'värväyttää',
        u'haluaisin',
        u'hyöyissä',
        u'saippuaa',
        u'touon',
        ]

    t1 = timeit(
        'timeit_test(WORDS, f=True)',
        setup='from __main__ import timeit_test, WORDS',
        number=1000)  # 0.300

    t2 = timeit(
        'timeit_test(WORDS)',
        setup='from __main__ import timeit_test, WORDS',
        number=1000,  # 0.585
        )

    print t1
    print t2
