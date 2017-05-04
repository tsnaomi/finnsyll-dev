# coding=utf-8

from os import sys, path

# import local finnsyll project with unreleased developments;
# otherwise, import pip installed finnsyll package
projects = path.dirname(path.dirname(path.abspath(__file__)))
sys.path = [path.join(projects, 'finnsyll'), ] + sys.path

from finnsyll import FinnSyll, phonology as phon  # noqa

StressedFinnSyll = FinnSyll(rules=True, stress=True)
_FinnSyll = FinnSyll(rules=False)
FinnSyll = FinnSyll(rules=True)
