# coding=utf-8

from phonology import replace_umlauts


def delimit(word):
    '''Insert syllable breaks at non-delimited compound boundaries.'''
    return word


if __name__ == '__main__':

    # WRITE TESTS

    words = []

    for word in words:
        print replace_umlauts(delimit(replace_umlauts(word)), put_back=True)
