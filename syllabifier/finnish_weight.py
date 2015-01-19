# coding=utf-8
from finnish_functions import *  # TODO


# return the syllable weight of a single syllable
def syll_weight(syll_split):

    # if the nucleus is long, heaviest
    if len(syll_split[Syllable.nucleus]) > 1:
        return Weight.CVV

    # if a coda is present, heavy
    elif len(syll_split[Syllable.coda]) > 0:
        return Weight.CVC

    else:
        return Weight.CV  # light


# given a list of syllables, store their weights in weights
def make_weights(syllables):

    weights = []

    for syll in syllables:
        weights += [syll_weight(syll)]

    return weights
