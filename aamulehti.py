# coding=utf-8

import finnsyll as finn
import os
import sys
import xml.etree.ElementTree as ET

from collections import Counter, namedtuple

# word forms: 991730 (exluding unseen lemmas)
# xml files: 61,529


# Tokens ----------------------------------------------------------------------

characters = u'aAbBcCDdeEfFGgHhiIjJkKLlMmnNoOpPqQRrSsTtUuVvWwxXyYzZ -äöÄÖ'
invalid_types = ['Delimiter', 'Abbrev', 'Code']

Token = namedtuple('Token', ['orth', 'lemma', 'msd', 'pos'])
TOKENS = []


def populate_db_tokens_from_aamulehti_1999():
    for tup in os.walk('aamulehti-1999'):
        dirpath, dirname, filenames = tup

        if dirpath == 'aamulehti-1999':
            continue

        for f in filenames:
            filepath = dirpath + '/' + f
            accumulate_tokens(f, filepath)

            break

        print dirpath

    global TOKENS
    FREQS = Counter(TOKENS)
    TOKENS = None  # save memory
    tokens = distill_tokens(FREQS)
    FREQS = None  # save memory

    save_tokens(tokens)

    print '%s tokens' % len(tokens)


def accumulate_tokens(filename, filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()

    try:
        for w in root.iter('w'):
            lemma = w.attrib['lemma']
            msd = w.attrib['msd']
            pos = w.attrib['type']
            t = w.text or ''

            if t != t.upper():

                if all([t, lemma, pos, pos not in invalid_types]):

                    orth = t.lower() if pos != 'Proper' else t
                    lemma = lemma.lower() if pos != 'Proper' else lemma

                    tok = Token(orth, lemma, msd, pos)
                    TOKENS.append(tok)

    except Exception as E:
        print filename, E


def distill_tokens(freqs):
    delete = []

    for t in freqs.iterkeys():
        if not all([1 if i in characters else 0 for i in t.orth]):  # isalpha()
            delete.append(t)

    for t in delete:
        del freqs[t]

    return freqs


def save_tokens(tokens):
    for token, freq in tokens.iteritems():
        t = finn.Token(token.orth, token.lemma, token.msd, token.pos, freq)
        finn.db.session.add(t)

    finn.db.session.commit()


# Documents -------------------------------------------------------------------

INDICES = {}


def populate_db_docs_from_aamulehti_1999():
    collect_token_ids()

    for tup in os.walk('aamulehti-1999'):
        dirpath, dirname, filenames = tup

        if dirpath == 'aamulehti-1999':
            continue

        for f in filenames:
            filepath = dirpath + '/' + f
            accumulate_docs(f, filepath)

            break

        print dirpath

    global INDICES
    INDICES = None  # save memory

    finn.syllabify_tokens()
    finn.db.session.commit()


def collect_token_ids():
    query = finn.Token.query.all()

    for t in query:
        tok, tok_id = (t.orth, t.lemma, t.msd, t.pos), t.id
        INDICES[tok] = tok_id


def accumulate_docs(filename, filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()

    tokens = set()
    tokenized_text = []

    try:
        for w in root.iter('w'):
            t = w.text or ''
            lemma = w.attrib['lemma']
            msd = w.attrib['msd']
            pos = w.attrib['type']

            # convert words that are not proper nouns into lowercase
            orth = t.lower() if pos != 'Proper' else t
            lemma = lemma.lower() if pos != 'Proper' else lemma

            tok = (orth, lemma, msd, pos)

            # find Token.id if one exists
            word = INDICES.get(tok, None)

            if word is not None:
                tokens.add(word)
                tokenized_text.append(word)

            # keep punctuation, acronyms, and numbers as strings
            else:
                tokenized_text.append(t)

        # create document instance
        doc = finn.Document(filename, tokenized_text, list(tokens))
        finn.db.session.add(doc)

    except Exception as E:
        print filename, E


# Lemmas ----------------------------------------------------------------------

def syllabify_unseen_lemmas():
    # get all unique lemmas
    lemmas = finn.db.session.query(finn.Token.lemma, finn.Token.pos).distinct()

    # isolate lemmas that do not have their own Tokens
    unseen = [t for t in lemmas if not finn.find_token(t[0].replace('_', ' '))]

    print '%s unseen lemmas' % len(unseen)
    import pdb
    pdb.set_trace()

    # create Tokens for unseen lemmas
    for t in unseen:
        word = finn.Token(t[0].replace('_', ' '), t[0], '', t[1], 0)
        word.syllabify()
        finn.db.session.add(word)
        finn.db.session.commit()


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # populate_db_tokens_from_aamulehti_1999()  # 13114.48 seconds
    # populate_db_docs_from_aamulehti_1999()  # 4221.7 seconds
    # syllabify_unseen_lemmas()
    finn.transition(pdf='--pdf' in sys.argv)
