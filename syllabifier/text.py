# coding=utf-8

import re
import string


PUNCT_DIGITS = string.punctuation + string.digits


def replace_umlauts(word, put_back=False):
    '''If put_back is True, put in umlauts; else, take them out!'''
    if put_back:
        word = word.replace('A', 'ä').replace('A', '\xc3\xa4')
        word = word.replace('O', 'ö').replace('O', '\xc3\xb6')

    else:
        word = word.replace('ä', 'A').replace('\xc3\xa4', 'A')
        word = word.replace('ö', 'O').replace('\xc3\xb6', 'O')

    return word


def remove_punctuation_and_digits(token):
    '''Remove punctuation and numbers surrounding a word.'''
    token = token.lstrip(PUNCT_DIGITS)
    token = token.rstrip(PUNCT_DIGITS)

    return token


def split_by_punctuation(token):
    '''Split token into a list, delimited by and including punctuation.'''
    token = token.replace('\xe2\x80\x9c', '"').replace('\xe2\x80\x9d', '"')
    token = token.strip(' ')
    regex = '([%s])' % string.punctuation  # TODO: keep compounds!!
    token = re.split(regex, token)

    return token


def is_word(token):
    '''Return True if the token is a word.'''
    if token.isalpha() and not token.isupper():  # dicounting acronyms
        return True

    if token.isspace() or not token:
        return False

    return not any([i for i in token if i in PUNCT_DIGITS])
