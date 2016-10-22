# coding=utf-8

from os import sys, path

# import local finnsyll project with unreleased developments;
# otherwise, import pip installed finnsyll package
projects = path.dirname(path.dirname(path.abspath(__file__)))
sys.path = [path.join(projects, 'finnsyll'), ] + sys.path

from finnsyll import FinnSyll

FinnSyll = FinnSyll()
