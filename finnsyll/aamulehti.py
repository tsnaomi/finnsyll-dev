# coding=utf-8

import finnsyll
import os
import xml.etree.ElementTree as ET


invalid_types = ['Delimiter', 'Abbrev', 'Numeral']


def populate_db_from_aamulehti_1999():
    for tup in os.walk('../aamulehti-1999'):
        dirpath, dirname, filenames = tup

        if dirpath == '../aamulehti-1999':
            continue

        for f in filenames:
            filepath = dirpath + '/' + f
            decode_xml_file(f, filepath)
            return

    # syllabify_unseen_lemmas()


def decode_xml_file(filename, filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()

    tokens = []
    tokenized_text = []

    try:

        for w in root.iter('w'):
            attrs = w.attrib
            t = w.text

            # ignore punctuation and acronyms
            if attrs['type'] not in invalid_types:

                # comvert words that are not proper nouns into lowercase
                t = t.lower() if attrs['type'] != 'Proper' else t

                # convert lemmas that are not proper nouns into lowercase
                l = attrs['lemma']
                l = l.lower() if attrs['type'] != 'Proper' else l

                word = finnsyll.find_token(t)

                # create token for word if one does not already exist
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

            # keep punctuation and acronyms as strings
            else:
                tokenized_text.append(t)

        # create document instance
        doc = finnsyll.Document(filename, tokens, tokenized_text)
        finnsyll.db.session.add(doc)

        finnsyll.db.session.commit()

    except Exception as E:
        print E
        import pdb; pdb.set_trace()
        print
        print
        print


def syllabify_unseen_lemmas():
    # get all unique lemmas
    tokens = finnsyll.Token.query.all()
    lemmas = [t.lemma for t in tokens]
    lemmas = list(set(lemmas))

    for t in lemmas:
        lemma = finnsyll.find_token(t.lemma)

        # create token for lemma if one does not already exist
        if not lemma:
            word = finnsyll.Token(t.lemma)
            word.lemma = t.lemma
            word.pos = t.pos
            finnsyll.db.session.add(word)
            finnsyll.db.session.commit()


if __name__ == '__main__':
    populate_db_from_aamulehti_1999()
