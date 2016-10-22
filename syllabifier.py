# coding=utf-8

from os import sys, path

try:
    # import local finnsyll project with unreleased developments
    projects = path.dirname(path.dirname(path.abspath(__file__)))
    sys.path = [path.join(projects, 'finnsyll'), ] + sys.path
    from finnsyll import FinnSyll

except Exception as e:
    # import pip installed finnsyll package
    from finnsyll import FinnSyll

FinnSyll = FinnSyll()
