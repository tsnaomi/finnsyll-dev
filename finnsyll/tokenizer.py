# coding=utf-8

import finnsyll
import pprint
import re
import string
import sys


PUNCT_DIGITS = string.punctuation + string.digits


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
    if token.isalpha() and not token.isupper():  # exclude acronyms
        return True

    if token.isspace() or not token:
        return False

    return not any([i for i in token if i in PUNCT_DIGITS])


def tokenize(filename):
    try:
        f = open(filename, 'r')
        text = f.readlines()
        f.close()

        token_IDs = []
        pickled_text = []

        for line in text:
            line = line.split(' ')

            for i in line:
                tokens = split_by_punctuation(i)

                for t in tokens:

                    if is_word(t):
                        word = finnsyll.find_token(t)

                        if not word:
                            word = finnsyll.Token(t)
                            finnsyll.db.session.add(word)
                            finnsyll.db.session.commit()

                        token_IDs.append(word.id)
                        pickled_text.append(word.id)

                    elif t:
                        pickled_text.append(t)

        return text, token_IDs, pickled_text

    except IOError:
        raise IOError('File %s could not be opened.' % filename)


# This is for seeing how well we're tokenizing text files. The function
# _tokenize() takes a text file, and prints two lists into the command line:
# a list of the words contained in the file, and a complete list of the words,
# punctuation marks, and numbers contained in the file. A version of pickle()
# is utilized in finnsyll.py.

def _tokenize(filename='fin.txt'):
    try:
        f = open(filename, 'r')
        text = f.readlines()
        f.close()

        tokenized_IDs = []
        tokenized_text = []

        for line in text:
            line = line.split(' ')

            for i in line:
                tokens = split_by_punctuation(i)

                for t in tokens:

                    if is_word(t):
                        tokenized_IDs.append(t)

                    if t:
                        tokenized_text.append(t)

        pprint.pprint(tokenized_IDs, width=1)
        pprint.pprint(tokenized_text, width=1)

    except IOError:
        raise IOError('File %s could not be opened.' % filename)


if __name__ == '__main__':
    args = sys.argv[1:]

    if args:

        if len(args) > 1 and args[1] == '-d':
            doc = finnsyll.Document(args[0])
            finnsyll.db.session.add(doc)
            finnsyll.db.session.commit()

        else:
            _tokenize(args[0])

    else:
        _tokenize()
