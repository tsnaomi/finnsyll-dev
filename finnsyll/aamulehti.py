# coding=utf-8

import finnsyll
import os
import xml.etree.ElementTree as ET


# word forms estimate: 986000 (exluding unseen lemmas)

characters = u'aAbBcCDdeEfFGgHhiIjJkKLlMmnNoOpPqQRrSsTtUuVvWwxXyYzZ -äöÄÖ'
invalid_types = ['Delimiter', 'Abbrev', 'Code']


def isalpha(word):
    return all([1 if i in characters else 0 for i in word])


def find_token(orth, lemma=None, msd=None, pos=None):
    '''Retrieve token by its orthography, lemma, msd, and pos.'''
    try:
        # ilike queries are case insensitive
        token = finnsyll.Token.query.filter(finnsyll.Token.orth.ilike(orth)).\
            filter_by(lemma=lemma).filter_by(msd=msd).\
            filter_by(pos=pos).first()

        return token

    except KeyError:
        return None


def populate_db_from_aamulehti_1999():
    for tup in os.walk('../aamulehti-1999'):
        dirpath, dirname, filenames = tup

        if dirpath == '../aamulehti-1999':
            continue

        for f in filenames:
            filepath = dirpath + '/' + f
            decode_xml_file(f, filepath)

        print dirpath

        break  # TODO

    print '%s tokens' % finnsyll.Token.query.count()


def decode_xml_file(filename, filepath, words):
    tree = ET.parse(filepath)
    root = tree.getroot()

    tokens = set()
    tokenized_text = []

    try:
        for w in root.iter('w'):
            lemma = w.attrib['lemma']
            msd = w.attrib['msd']
            pos = w.attrib['type']
            t = w.text or ''

            # ignore null lemmas, null types, and illegal characters and types
            if all([t, lemma, msd, pos, isalpha(t), pos not in invalid_types]):

                # convert words that are not proper nouns into lowercase
                t = t.lower() if pos != 'Proper' else t
                lemma = lemma.lower() if pos != 'Proper' else lemma

                word = find_token(t, lemma=lemma, msd=msd, pos=pos)

                # create Token for word if one does not already exist
                if not word:
                    word = finnsyll.Token(t)
                    word.msd = msd
                    word.pos = pos
                    word.lemma = lemma.replace('_', ' ')

                # update the word's frequency count
                word.freq += 1

                finnsyll.db.session.add(word)
                finnsyll.db.session.commit()

                tokens.add(word.id)
                tokenized_text.append(word.id)

            # keep punctuation, acronyms, and numbers as strings
            else:
                tokenized_text.append(t)

        # create document instance
        doc = finnsyll.Document(filename, list(tokens), tokenized_text)
        finnsyll.db.session.add(doc)

        finnsyll.db.session.commit()

    except Exception as E:
        print filename, E

    return words


def syllabify_unseen_lemmas():
    # get all unique lemmas
    tokens = finnsyll.Token.query.all()
    lemmas = [(t.lemma.replace('_', ' '), t.pos) for t in tokens]
    lemmas = list(set(lemmas))

    # isolate lemmas that do not have their own Tokens
    unseen_lemmas = [t for t in lemmas if not finnsyll.find_token(t[0])]

    print '%s unseen lemmas' % len(unseen_lemmas)
    import pdb
    pdb.set_trace()

    # create Tokens for unseen lemmas
    for t in unseen_lemmas:
        word = finnsyll.Token(t[0])
        word.lemma = t[0]
        word.pos = t[1]
        finnsyll.db.session.add(word)
        finnsyll.db.session.commit()


if __name__ == '__main__':
    populate_db_from_aamulehti_1999()