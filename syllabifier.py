# coding=utf-8

from os import sys, path

# import local finnsyll project with unreleased developments;
# otherwise, import pip installed finnsyll package
projects = path.dirname(path.dirname(path.abspath(__file__)))
sys.path = [path.join(projects, 'finnsyll'), ] + sys.path

from finnsyll import FinnSyll, phonology as phon  # noqa

_FinnSyll = FinnSyll(track_rules=False)
FinnSyll = FinnSyll(track_rules=True)
