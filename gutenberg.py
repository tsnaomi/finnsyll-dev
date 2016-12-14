# coding=utf-8

import app
import os
import re

from datetime import datetime
from sqlalchemy.exc import IntegrityError

VOWELS = u'ieäyöauo'
CHARACTERS = u'abcdefghijklmnopqrstuvwxyz-äö'
PUNCTUATION = r'!"#\$%&\'()\*\+,\./:;<=>\?@\[\\\]\^_`{\|}~'


def extract_gutenberg():
    '''Extract poetry from Project Gutenberg.'''
    # wipe existing Gutenberg tokens prior to extractions
    wipe_gutenberg_tokens()

    # extract Gutenberg poetry
    for dirpath, dirname, filenames in os.walk('gutenberg/gutenberg'):

        for fn in filenames[1:]:
            fp = dirpath + '/' + fn
            Book, text = add_poet_and_book(fn, fp)
            add_sections(Book, text)


def wipe_gutenberg_tokens():
    '''Delete all non-Aamulehti tokens.'''
    app.Token.query.filter_by(is_aamulehti=False).delete()
    app.db.session.commit()


def add_poet_and_book(fn, fp):
    '''Add (or retrieve) Poet and Book objects.'''
    with open(fp, 'r') as f:
        f = list(f)

    header, text = f[:4], f[4:]
    header = ''.join(header).replace('\r', '')
    text = '\n'.join(re.split(  # too many blank lines...
        r'\r\n',
        re.sub(r'[^A-Z]\r\n\r\n\r\n', '\r\n\r\n', ''.join(text)),
        ))

    # add or get Poet
    surname = re.search(r'Author:.* ([A-Za-zÄÖäö]+)\n', header).group(1)
    try:
        Poet = app.Poet(surname=surname)
        app.db.session.add(Poet)
        app.db.session.commit()
    except IntegrityError:
        app.db.session.rollback()
        Poet = app.Poet.query.filter_by(surname=surname).one()

    # add Book
    title = re.search(r'Title: (.+)\n', header).group(1)
    Book = app.Book(title=title, poet_id=Poet.id)
    app.db.session.add(Book)
    app.db.session.commit()

    print '%s (%s)' % (Book.title, Poet.surname)

    return Book, text


def add_sections(Book, text):
    '''Add Section objects for Book.sections.'''
    # divide the book into sections
    sections = _divide_text(text)

    for i, section_text in enumerate(sections, start=1):

        # add Section
        Section = app.Section(section=i, book_id=Book.id)
        app.db.session.add(Section)
        app.db.session.commit()

        # tokenize text
        Section.text = _tokenize_text(section_text, Section)
        app.db.session.commit()


def _divide_text(text, n=500):
        '''Divide the book of poetry into sections of ~n lines.'''
        new_poem = r'(\n[A-ZÄÖ0-9\. ]+\n)'
        poems = re.split(new_poem, text)

        if len(poems) == 1:
            new_poem = r'(\n\n)'
            poems = re.split(new_poem, text)

        sections = []
        section = ''
        x = 0

        for poem in poems:
            if x > n and re.match(new_poem, poem):
                sections.append(section)
                section = ''
                x = 0

            x += poem.count('\n')
            section += poem

        sections.append(section)

        return sections


def _tokenize_text(section_text, Section, add_objects=True):
    '''Tokenize text for Section.text'''
    tokenized_text = []
    string = ''

    for line in section_text.split('\n'):

        # if the line is blank, insert an HTML breakpoint
        if not line:
            string += '<br>'
            continue

        string += '<div>'

        # split line by punctuation, spaces, and newline characters:
        # 'päälle pään on taivosehen;' >
        # ['päälle', ' ', 'pään', ' ', 'on', ' ', 'taivosehen', ';']
        line = filter(None, re.split(
            r'(\r\n|[ ]+|[%s]|--)' % PUNCTUATION,
            line,
            ))

        for word in line:
            word = word.decode('utf-8', errors='replace')

            # if the word is a series of spaces, insert HTML non-breaking
            # spaces of an equivalent length
            if len(word) > 1 and word == len(word) * ' ':
                string += '&nbsp;' * len(word)
                continue

            # ignore any words that appear in all uppercase (e.g., acronyms)
            if word == word.upper():
                string += word
                continue

            word = word.lower()

            # find all u- and y-final diphthongs in word
            sequences = _get_u_y_final_diphthongs(word.encode('utf-8'))

            # if the word contains a u- and y-final diphthong and is composed
            # of acceptable characters...
            if sequences and all(1 if i in CHARACTERS else 0 for i in word):
                tokenized_text.append(string)
                string = ''

                if add_objects:

                    # add Variant to db and tokenized_text
                    Variant = add_variant(word, Section)
                    tokenized_text.append(Variant.id)

                    # add Sequences
                    add_sequences(sequences, word, Variant)

                else:
                    tokenized_text.append(None)

            else:
                string += word

        string += '</div>'

    tokenized_text.append(string)  # WHOOPS

    return tokenized_text


def _get_u_y_final_diphthongs(word):
    '''Extract u- and y-final diphthongs from word.'''
    pattern = r'(?=([^aäoöieuy]{1}|^)(au|eu|ou|iu|iy|ey|äy|öy)([^aäoöieuy]{1}|$))'  # noqa
    sequences = list(re.finditer(pattern, word))

    return sequences


def add_variant(word, Section):
    '''Add Variant object for Section.variants.'''
    # find existing Token
    Token = app.find_token(word)

    # create a new Token if one does not already exist
    if not Token:
        Token = app.Token(orth=word)
        app.db.session.add(Token)

    Token.is_gutenberg = True
    app.db.session.commit()

    # add Variant
    Variant = app.Variant(token_id=Token.id, section_id=Section.id)
    app.db.session.add(Variant)
    app.db.session.commit()

    return Variant


def add_sequences(sequences, word, Variant):
    '''Add VV objects for Variant.sequences.'''
    for seq in sequences:
        i = seq.start(2)
        is_heavy, is_stressed, split = _get_phonotactics(seq, i, word)

        # add VV
        VV = app.VV(
            poet_id=Variant._section._book._poet.id,
            book_id=Variant._section._book.id,
            variant_id=Variant.id,
            sequence=seq.group(2),
            index=i,
            html=_get_html(seq, word),
            is_heavy=is_heavy,
            is_stressed=is_stressed,
            split=split,
            )
        app.db.session.add(VV)

    app.db.session.commit()


def _get_html(sequence, word):
    '''Create the html representation for word, emboldening sequence.'''
    vv = sequence.group(2).decode('utf-8')
    i = sequence.start(2)
    j = i + len(vv)
    html = '%s<strong>%s</strong>%s' % (word[:i], vv, word[j:])

    return html


def _get_phonotactics(sequence, i, word):
    '''Determine the weight and primary stress of the sequence's syllable.'''
    is_stressed = not any(v in VOWELS for v in word[:i])
    split = 'join' if is_stressed else None

    try:
        is_heavy = word[sequence.end(2) + 1] not in VOWELS
    except IndexError:
        is_heavy = word[-1] not in VOWELS

    return is_heavy, is_stressed, split


def populate_line():
    '''Populate VV.line for each VV sequence.'''
    sequences = app.VV.query.all()

    for VV in sequences:
        text = VV._variant._section.text
        index = text.index(VV._variant.id)
        try:
            pre = re.split(r'\n|</div>|<div>|<br>', text[index - 1])[-1]
        except IndexError:
            pre = ''
        try:
            post = re.split(r'\n|</div>|<div>|<br>', text[index + 1])[0]
        except IndexError:
            post = ''
        line = '%s%s%s' % (pre, VV.orth.upper(), post)
        line = line.replace('&nbsp;', ' ').replace('</strong></span>', '')
        line = line.replace("<span style='font-size:30px;'><strong><span style='font-size:1px;'>@</span>", '')  # noqa
        VV.line = line

    app.db.session.commit()


def fix_html_umlaut_bug():
    '''Fix VV.html for words where an umlaut precedes VV.sequence.'''
    def replace_umlauts(word, put_back=False):
        '''Temporarily replace umlauts with easier-to-read characters.'''
        if put_back:
            word = word.replace('A', u'ä').replace('A', u'\xc3\xa4')
            word = word.replace('O', u'ö').replace('O', u'\xc3\xb6')

        else:
            word = word.replace(u'ä', 'A').replace(u'\xc3\xa4', 'A')
            word = word.replace(u'ö', 'O').replace(u'\xc3\xb6', 'O')

        return word

    for vv in app.VV.query.all():

        if vv.html.replace('<strong>', '').replace('</strong>', '') != vv.orth:
            orth = replace_umlauts(vv.orth)
            html = replace_umlauts(vv.html)
            seq = re.search(r'<strong>(.{2})</strong>', html).groups()[0]
            i = html.index('<')
            i = i - orth[:i].count('A') - orth[:i].count('O')
            i += 1 if seq[0] in 'AO' else 0
            html = orth[:i] + '<strong>' + seq + '</strong>' + orth[i + 2:]
            vv.html = replace_umlauts(html, put_back=True)
            vv.index = vv.html.find('<')

    app.db.session.commit()


def fix_final_text_bug():
    '''Add the missing section-final strings to each section.'''
    for dirpath, dirname, filenames in os.walk('gutenberg/gutenberg'):

        for fn in filenames[1:]:

            with open(dirpath + '/' + fn, 'r') as f:
                f = list(f)

            header, text = f[:4], f[4:]
            header = ''.join(header).replace('\r', '')
            text = '\n'.join(re.split(  # too many blank lines...
                r'\r\n',
                re.sub(r'[^A-Z]\r\n\r\n\r\n', '\r\n\r\n', ''.join(text)),
                ))

            # get Poet
            surname = re.search(
                r'Author:.* ([A-Za-zÄÖäö]+)\n',
                header,
                ).group(1)
            Poet = app.Poet.query.filter_by(surname=surname).one()

            # get Book and section texts
            title = re.search(r'Title: (.+)\n', header).group(1)
            Book = app.Book.query.filter_by(title=title, poet_id=Poet.id).one()
            sections = _divide_text(text)

            for i, section_text in enumerate(sections, start=1):

                # get Section
                Section = app.Section.query.filter_by(
                    section=i,
                    book_id=Book.id,
                    ).one()

                # extract and save final the section-final text string
                tokenized_text = _tokenize_text(section_text, Section, False)
                final_text = tokenized_text[-1]

                if final_text and len(Section.text) != len(tokenized_text):
                    text = Section.text + [final_text, ]
                    Section.text = text

            app.db.session.commit()


if __name__ == '__main__':
    print datetime.utcnow()
    # extract_gutenberg()
    # populate_line()
    # fix_html_umlaut_bug()
    # fix_final_text_bug()
    print datetime.utcnow()
