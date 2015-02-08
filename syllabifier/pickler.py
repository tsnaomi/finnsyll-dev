# coding=utf-8

import sys

from pprint import pprint
from ..text import is_word, split_by_punctuation

# This file is for seeing how well we're "pickling" text files. The function
# pickle() takes a text file, and prints two lists into the command line:
# a list of the words contained in the file, and a complete list of the words,
# punctuation marks, and numbers contained in the file. A version of pickle()
# is utilized in finnsyll.py.

pickled_IDs = []
pickled_text = []


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
