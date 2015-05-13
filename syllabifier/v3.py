# coding=utf-8

import re

from phonology import (
    contains_diphthong,
    contains_VV,
    contains_Vu_diphthong,
    contains_Vy_diphthong,
    contains_VVV,
    is_consonant,
    is_long,
    is_vowel,
    replace_umlauts,
    VOWELS,
    )


# Syllabifier -----------------------------------------------------------------

def syllabify(word):
    compound = True if '-' in word or ' ' in word else None

    # if the word contains a delimiter (a hyphens or space), split the word
    # along the delimiter(s) and syllabify the individual parts separately
    if compound:
        WORD, RULES = [], []

        for w in re.split('(-| )', word):
            if w in '- ':
                syll, rules = w, ' |'

            else:
                syll, rules = _syllabify(w)

            WORD.append(syll)
            RULES.append(rules)

        return ''.join(WORD), ''.join(RULES)

    return _syllabify(word)


def _syllabify(word):
    '''Syllabify the given word.'''
    word = replace_umlauts(word)
    word, applied_rules = apply_T1(word)

    if re.search(r'[^ieAyOauo]*([ieAyOauo]{2})[^ieAyOauo]*', word):
        word, T2 = apply_T2(word)
        word, T8 = apply_T8(word)
        word, T4 = apply_T4(word)
        applied_rules += T2 + T8 + T4
        # applied_rules += T2 + T4

    if re.search(r'[ieAyOauo]{3}', word):
        word, T5 = apply_T5(word)
        word, T6 = apply_T6(word)
        word, T7 = apply_T7(word)
        applied_rules += T5 + T6 + T7

    word = replace_umlauts(word, put_back=True)

    return word, applied_rules


# T1 --------------------------------------------------------------------------

def apply_T1(word):
    '''There is a syllable boundary in front of every CV-sequence.'''
    WORD = _split_consonants_and_vowels(word)

    for i, v in enumerate(WORD):

        if i == 0 and is_consonant(v[0][0]):
            continue

        elif is_consonant(v[0]) and i + 1 != len(WORD):
            WORD[i] = v[:-1] + '.' + v[-1]

    word = ''.join(WORD)
    RULE = ' T1'

    return word, RULE


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
    vowels that are not a genuine diphthong, e.g., [ta.e], [ko.et.taa].'''
    WORD = word.split('.')

    for i, v in enumerate(WORD):

        if not contains_diphthong(v):
            VV = contains_VV(v)

            if VV and not is_long(VV.group(1)):
                I = VV.start(1) + 1
                WORD[i] = v[:I] + '.' + v[I:]

    WORD = '.'.join(WORD)
    RULE = ' T2' if word != WORD else ''

    return WORD, RULE


# T4 --------------------------------------------------------------------------

def apply_T4(word):
    '''An agglutination diphthong that ends in /u, y/ usually contains a
    syllable boundary when -C# or -CCV follow, e.g., [lau.ka.us],
    [va.ka.ut.taa].'''
    WORD = word.split('.')

    for i, v in enumerate(WORD):

        # i % 2 != 0 prevents this rule from applying to first, third, etc.
        # syllables, which receive stress (WSP)
        if is_consonant(v[-1]) and i % 2 != 0:

            if i + 1 == len(WORD) or is_consonant(WORD[i + 1][0]):
                VV = contains_Vu_diphthong(v) or contains_Vy_diphthong(v)

                if VV:
                    I = VV.start(1) + 1
                    WORD[i] = v[:I] + '.' + v[I:]

    WORD = '.'.join(WORD)
    RULE = ' T4' if word != WORD else ''

    return WORD, RULE


# T5 --------------------------------------------------------------------------

i_DIPHTHONGS = ['ai', 'ei', 'oi', 'Ai', 'Oi', 'ui', 'yi']


def apply_T5(word):  # BROKEN
    '''If a (V)VVV-sequence contains a VV-sequence that could be an /i/-final
    diphthong, there is a syllable boundary between it and the third vowel,
    e.g., [raa.ois.sa], [huo.uim.me], [la.eis.sa], [sel.vi.äi.si], [tai.an],
    [säi.e], [oi.om.me].'''
    WORD = word.split('.')

    for i, v in enumerate(WORD):
        if contains_VVV(v) and any(i for i in i_DIPHTHONGS if i in v):
            I = v.rfind('i') - 1 or 2
            I = I + 2 if is_consonant(v[I - 1]) else I
            WORD[i] = v[:I] + '.' + v[I:]

    WORD = '.'.join(WORD)
    RULE = ' T5' if word != WORD else ''

    return WORD, RULE


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

                if i != 0 and is_vowel(v[I - 1]):  # e.g., tuaa
                    WORD[i] = v[:I] + '.' + v[I:]

                else:  # e.g., viia
                    WORD[i] = v[:I + 2] + '.' + v[I + 2:]

    WORD = '.'.join(WORD)
    RULE = ' T6' if word != WORD else ''

    return WORD, RULE


# T7 --------------------------------------------------------------------------

def apply_T7(word):
    '''If a VVV-sequence does not contain a potential /i/-final diphthong,
    there is a syllable boundary between the second and third vowels, e.g.
    [kau.an], [leu.an], [kiu.as].'''
    WORD = word.split('.')

    for i, v in enumerate(WORD):

        if contains_VVV(v):

            for I, V in enumerate(v[::-1]):

                if is_vowel(V):
                    WORD[i] = v[:I] + '.' + v[I:]

    WORD = '.'.join(WORD)
    RULE = ' T7' if word != WORD else ''

    return WORD, RULE


# T8 --------------------------------------------------------------------------

def ie_sequences(word):
    # this regex pattern searches for /ie/ sequences that do not occur in the
    # first syllable, and that are not directly preceded or followed by a vowel
    return re.finditer(r'(?=\.[^ieAyOauo]*(ie)[^ieAyOauo]*($|\.))', word)


def apply_T8(word):
    '''Split /ie/ sequences in syllables that do not take primary stress.'''
    WORD = word
    offset = 0

    for ie in ie_sequences(WORD):
        i = ie.start(1) + 1 + offset
        WORD = WORD[:i] + '.' + WORD[i:]
        offset += 1

    RULE = ' T8' if word != WORD else ''

    return WORD, RULE


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    args = sys.argv[1:]

    if args:
        for arg in args:
            if isinstance(arg, str):
                print syllabify(arg) + '\n'

    else:
        words = [
            (u'tae', u'ta.e'),
            (u'koettaa', u'ko.et.taa'),
            (u'hain', u'hain'),  # ha.in (alternative)
            (u'laukaus', u'lau.ka.us'),
            (u'vakauttaa', u'va.ka.ut.taa'),
            (u'raaoissa', u'raa.ois.sa'),
            (u'huouimme', u'huo.uim.me'),
            (u'laeissa', u'la.eis.sa'),
            (u'selviäisi', u'sel.vi.äi.si'),
            (u'taian', u'tai.an'),
            (u'säie', u'säi.e'),
            (u'oiomme', u'oi.om.me'),
            (u'korkeaa', u'kor.ke.aa'),
            (u'yhtiöön', u'yh.ti.öön'),
            (u'ruuan', u'ruu.an'),
            (u'määytte', u'mää.yt.te'),
            (u'kauan', u'kau.an'),
            (u'leuan', u'leu.an'),
            (u'kiuas', u'kiu.as'),
            (u'haluaisin', u'ha.lu.ai.sin'),
            (u'hyöyissä', u'hyö.yis.sä'),
            (u'pamaushan', u'pa.ma.us.han'),
            (u'saippuaa', u'saip.pu.aa'),
            (u'joissa', u'jois.sa'),  # jo.is.sa (alternative)
            (u'tae', u'ta.e'),
            (u'kärkkyä', u'kärk.ky.ä'),
            (u'touon', u'tou.on'),
            (u'värväytyä', u'vär.väy.ty.ä'),
            (u'värväyttää', u'vär.vä.yt.tää'),
            (u'daniel', u'da.ni.el'),
            # (u'esseisti', u'es.se.is.ti'),
            # (u'evankeliumeja', u'e.van.ke.li.u.me.ja'),
            (u'sosiaalinen', u'so.si.aa.li.nen'),
            (u'välierien', u'vä.li.e.ri.en'),
            (u'lounais-suomen puhelin', u'lou.nais-suo.men pu.he.lin'),
            ]

        for word in words:
            syll = syllabify(word[0])

            if syll[0] != word[1]:
                print u'TRY: %s  %s\nYEA: %s\n' % (syll[0], syll[1], word[1])
                # print u'TEST: %s\nGOLD: %s\n' % (syll[0], word[1])
