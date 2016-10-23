# coding=utf-8

from os import sys, path

# import local finnsyll project with unreleased developments;
# otherwise, import pip installed finnsyll package
projects = path.dirname(path.dirname(path.abspath(__file__)))
sys.path = [path.join(projects, 'finnsyll'), ] + sys.path

from finnsyll import FinnSyll

FinnSyll = FinnSyll(track_rules=True)


def replace_umlauts(word, put_back=False):  # use translate()
    '''If put_back is True, put in umlauts; else, take them out!'''
    if put_back:
        word = word.replace(u'A', u'ä').replace(u'A', u'\xc3\xa4')
        word = word.replace(u'O', u'ö').replace(u'O', u'\xc3\xb6')

    else:
        word = word.replace(u'ä', u'A').replace(u'\xc3\xa4', u'A')
        word = word.replace(u'ö', u'O').replace(u'\xc3\xb6', u'O')

    return word
