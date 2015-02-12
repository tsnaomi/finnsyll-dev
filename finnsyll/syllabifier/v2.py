# coding=utf-8

from phonology import (
    contains_diphthong,
    contains_VV,
    contains_Vu_diphthong,
    contains_Vy_diphthong,
    contains_VVV,
    is_consonant,
    is_consonantal_onset,
    is_vowel,
    replace_umlauts,
    VOWELS
    )


# Syllabifier -----------------------------------------------------------------

def syllabify(word):
    '''Syllabify the given word.'''
    word = replace_umlauts(word)
    word, CONTINUE_VV, CONTINUE_VVV = apply_T1(word)

    if CONTINUE_VV:
        word = apply_T2(word)
        word = apply_T4(word)

    if CONTINUE_VVV:
        word = apply_T5(word)
        word = apply_T6(word)
        word = apply_T7(word)

    word = replace_umlauts(word, put_back=True)

    return word


# T1 --------------------------------------------------------------------------

def apply_T1(word):
    '''There is a syllable boundary in front of every CV-sequence.'''
    WORD = _split_consonants_and_vowels(word)
    CONTINUE_VV = 0
    CONTINUE_VVV = 0

    for i, v in enumerate(WORD):

        if i == 0 and is_consonantal_onset(v):
            continue

        elif is_consonant(v[0]) and i + 1 != len(WORD):
            WORD[i] = v[:-1] + '.' + v[-1]

        elif is_vowel(v[0]):

            if len(v) > 2:
                CONTINUE_VVV += 1

            elif len(v) > 1:
                CONTINUE_VV += 1

    word = ''.join(WORD)

    return word, CONTINUE_VV, CONTINUE_VVV


def _same_syllabic_feature(ch1, ch2):
    # returns True if ch1 and ch2 are both vowels or both consonants
    # assumes either both ch1 and ch2 are either C or V
    ch1 = 'V' if ch1 in VOWELS else 'C'
    ch2 = 'V' if ch2 in VOWELS else 'C'

    return ch1 == ch2


def _split_consonants_and_vowels(word):
    # 'balloon' -> {1: 'b', 2: 'a', 3: 'll', 4: 'oo', 5: 'n'}
    # 'bal.loon' -> {1: 'b', 2: 'a', 3: 'l', 4: '.'. 5: 'l', 6: 'oo', 7: 'n'}
    WORD = []

    prev = [0, 0]  # (list indice, character)

    for ch in word:

        if prev[0] and _same_syllabic_feature(prev[1], ch):
            WORD[prev[0] - 1] += ch

        else:
            WORD.append(ch)
            prev[0] += 1
            prev[1] = ch

    return WORD


# T2 --------------------------------------------------------------------------

def apply_T2(word):
    '''There is a syllable boundary within a sequence VV of two nonidentical
    that are not a genuine diphthong, e.g., [ta.e], [ko.et.taa].'''
    WORD = word.split('.')

    for i, v in enumerate(WORD):

        if not contains_diphthong(v):
            VV = contains_VV(v)

            if VV:
                I = v.find(VV) + 1
                WORD[i] = v[:I] + '.' + v[I:]

    word = '.'.join(WORD)

    return word


# T4 --------------------------------------------------------------------------

def apply_T4(word):
    '''An agglutination diphthong that ends in /u, y/ usually contains a
    syllable boundary when -C# or -CCV follow, e.g., [lau.ka.us],
    [va.ka.ut.taa].'''
    WORD = word.split('.')

    for i, v in enumerate(WORD):

        if is_consonant(v[-1]):

            if i + 1 == len(WORD) or is_consonant(WORD[i + 1][0]):

                if contains_Vu_diphthong(v):
                    I = v.rfind('u')
                    WORD[i] = v[:I] + '.' + v[I:]

                elif contains_Vy_diphthong(v):
                    I = v.rfind('y')
                    WORD[i] = v[:I] + '.' + v[I:]

    word = '.'.join(WORD)

    return word


# T5 --------------------------------------------------------------------------

i_DIPHTHONGS = ['ai', 'ei', 'oi', 'Ai', 'Oi', 'ui', 'yi']


def apply_T5(word):
    '''If a (V)VVV-sequence contains a VV-sequence that could be an /i/-final
    diphthong, there is a syllable boundary between it and the third vowel,
    e.g., [raa.ois.sa], [huo.uim.me], [la.eis.sa], [sel.vi.äi.si], [tai.an],
    [säi.e], [oi.om.me].'''
    WORD = word.split('.')

    for i, v in enumerate(WORD):

        if contains_VVV(v) and any(i for i in i_DIPHTHONGS if i in v):
            I = v.rfind('i') - 1
            WORD[i] = v[:I] + '.' + v[I:]

    word = '.'.join(WORD)

    return word


# T6 --------------------------------------------------------------------------

LONG_VOWELS = [i + i for i in VOWELS]


def apply_T6(word):
    '''If a VVV-sequence contains a long vowel, there is a syllable boundary
    between it and the third vowel, e.g. [kor.ke.aa], [yh.ti.öön], [ruu.an],
    [mää.yt.te].'''
    WORD = word.split('.')

    for i, v in enumerate(WORD):

        if contains_VVV(v):
            VV = [v.find(j) for j in LONG_VOWELS if v.find(j) > 0]

            if VV:
                I = VV[0]

                if I + 2 == len(v) or is_vowel(v[I + 2]):
                    WORD[i] = v[:I] + '.' + v[I:]

                else:
                    WORD[i] = v[:I + 1] + '.' + v[I + 1:]  # TODO

    word = '.'.join(WORD)

    return word


# T7 --------------------------------------------------------------------------

def apply_T7(word):
    '''If a VVV-sequence does not contain a potential /i/-final diphthong,
    there is a syllable boundary between the second and third vowels, e.g.
    # [kau.an], [leu.an], [kiu.as].'''
    WORD = word.split('.')

    for i, v in enumerate(WORD):

        if contains_VVV(v):

            for I, V in enumerate(v[::-1]):

                if is_vowel(V):
                    WORD[i] = v[:I] + '.' + v[I:]

    word = '.'.join(WORD)

    return word


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    args = sys.argv[1:]

    if args:
        for arg in args:
            if isinstance(arg, str):
                print syllabify(arg) + '\n'

    else:
        # test syllabifications -- from Arto's finnish_syllabification.txt
        words = [
            (u'kala', u'ka.la'),  # T-1
            (u'järjestäminenkö', u'jär.jes.tä.mi.nen.kö'),  # T-1, 1, 1, 1, 1
            (u'kärkkyä', u'kärk.ky.ä'),  # T-1, 2
            (u'värväytyä', u'vär.väy.ty.ä'),  # T-1, 1, 2
            (u'pamaushan', u'pa.ma.us.han'),  # T-1, 4, 1
            (u'värväyttää', u'vär.vä.yt.tää'),  # T-1, 4, 1
            (u'haluaisin', u'ha.lu.ai.sin'),  # T-1, 5
            (u'hyöyissä', u'hyö.yis.sä'),  # T-5, 1
            (u'saippuaa', u'saip.pu.aa'),  # T-1, 6
            (u'touon', u'tou.on'),  # T-7
            ]

        for word in words:
            print u'TRY: %s\nYEA: %s\n' % (syllabify(word[0]), word[1])
