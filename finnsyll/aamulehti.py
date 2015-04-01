# coding=utf-8

import finnsyll
import os
import string
import xml.etree.ElementTree as ET


letters = string.letters + u'-äöÄÖ'
invalid_types = ['Delimiter', 'Abbrev']


def populate_db_from_aamulehti_1999():
    for tup in os.walk('../aamulehti-1999'):
        dirpath, dirname, filenames = tup

        if dirpath == '../aamulehti-1999':
            continue

        for f in filenames:
            filepath = dirpath + '/' + f
            decode_xml_file(f, filepath)
            return  # TODO

    syllabify_unseen_lemmas()


def decode_xml_file(filename, filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()

    tokens = []
    tokenized_text = []

    try:

        for w in root.iter('w'):
            attrs = w.attrib
            t = w.text

            # ignore punctuation, acronyms, and numbers
            if attrs['type'] not in invalid_types and isalpha(t):

                # convert words that are not proper nouns into lowercase
                t = t.lower() if attrs['type'] != 'Proper' else t

                # convert lemmas that are not proper nouns into lowercase
                l = attrs['lemma']
                l = l.lower() if attrs['type'] != 'Proper' else l

                word = finnsyll.find_token(t)

                # create Token for word if one does not already exist
                if not word or word.lemma != attrs['lemma'] or \
                        word.msd != attrs['msd'] or word.pos != attrs['type']:
                    word = finnsyll.Token(t)
                    word.msd = attrs['msd']
                    word.pos = attrs['type']
                    word.lemma = l

                # update the word's frequency count
                word.freq += 1

                finnsyll.db.session.add(word)
                finnsyll.db.session.commit()

                tokens.append(word.id)
                tokenized_text.append(word.id)

            # keep punctuation, acronyms, and numbers as strings
            else:
                tokenized_text.append(t)

        # create document instance
        doc = finnsyll.Document(filename, tokens, tokenized_text)
        finnsyll.db.session.add(doc)

        finnsyll.db.session.commit()

    except Exception as E:
        print filename, E


def syllabify_unseen_lemmas():
    # get all unique lemmas
    tokens = finnsyll.Token.query.all()
    lemmas = [(t.lemma, t.pos) for t in tokens]
    lemmas = list(set(lemmas))

    for t in lemmas:
        lemma = finnsyll.find_token(t[0])

        # create Token for lemma if one does not already exist
        if not lemma:
            word = finnsyll.Token(t[0])
            word.lemma = t[0]
            word.pos = t[1]
            finnsyll.db.session.add(word)
            finnsyll.db.session.commit()


def isalpha(word):
    return all([1 if i in letters else 0 for i in word])


if __name__ == '__main__':
    populate_db_from_aamulehti_1999()
