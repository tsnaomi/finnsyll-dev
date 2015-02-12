# coding=utf-8

from timeit import timeit
# from syllabifier.phonology import (
#     annotate_stress,
#     annotate_weight,
#     get_sonorities,
#     get_stress_pattern,
#     get_weights
#     )
from syllabifier.falk.finnish_syllables import make_syllables
from syllabifier.v2 import syllabify

# class Word(object):
#     syllabified = None
#     weights = None
#     sonorities = None
#     stress = None

#     def __init__(self, orthography):
#         self.syllabified = syllabify(orthography)

#         syllables = self.syllabified.split('.')
#         sonorities = get_sonorities(syllables)
#         weights = get_weights(syllables)
#         stress = get_stress_pattern(weights)[0]

#         STRESS = [annotate_stress(i) for i in stress]
#         WEIGHTS = [annotate_weight(i) for i in weights]

#         self.stress = ''.join(STRESS)
#         self.sonorities = ''.join(sonorities)
#         self.weights = ''.join(WEIGHTS)

#     def __repr__(self):
#         rep = 'Syllabification: %s' % self.syllabified
#         rep += '\n\tStress: %s' % self.stress
#         rep += '\n\tSonority: %s' % self.sonorities
#         rep += '\n\tWeight: %s\n' % self.weights

#         return rep

#     def __unicode__(self):
#         return self.__repr__()


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
