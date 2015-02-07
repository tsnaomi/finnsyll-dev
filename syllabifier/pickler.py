# coding=utf-8

import sys
import string
import re

from pprint import pprint

# This file is for seeing how well we're "pickling" text files. The function
# pickle() takes a text file, and prints two lists into the command line:
# a list of the words contained in the file, and a complete list of the words,
# punctuation marks, and numbers contained in the file. A version of pickle()
# is utilized in finnsyll.py.

PUNCT_DIGITS = string.punctuation + string.digits

pickled_IDs = []
pickled_text = []


def is_word(token):
    '''Return True if the token is a word.'''
    if token.isalpha():
        return True

    if token.isspace() or not token:
        return False

    return not any([i for i in token if i in PUNCT_DIGITS])


def remove_punctuation_and_digits(token):
    '''Remove punctuation and numbers surrounding a word.'''
    token = token.lstrip(PUNCT_DIGITS)
    token = token.rstrip(PUNCT_DIGITS)

    return token


def split_by_punctuation(token):
    '''Split token into a list, delimited by and including punctuation.'''
    token = token.replace('\xe2\x80\x9c', '"').replace('\xe2\x80\x9d', '"')
    token = token.strip(' ')
    regex = '([%s])' % string.punctuation
    token = re.split(regex, token)

    return token


def pickle(filename):
    if len(filename) != 1:
        raise ValueError('Please enter a single file to annotate.')

    try:
        f = open(filename, 'r')
        text = f.readlines()
        f.close()

        for line in text:
            line = line.split(' ')

            for i in line:
                tokens = split_by_punctuation(i)

                for t in tokens:

                    if is_word(t):
                        pickled_IDs.append(t)

                    if t:
                        pickled_text.append(t)

        pprint(pickled_IDs, width=1)
        pprint(pickled_text, width=1)

    except IOError:
        raise IOError('File %s could not be opened.' % filename)


if __name__ == '__main__':
    args = sys.argv[1:]
    pickle(args)
