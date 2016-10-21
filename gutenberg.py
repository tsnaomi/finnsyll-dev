# coding=utf-8

import app as finn
import os
import re

from datetime import datetime

characters = u'abcdefghijklmnopqrstuvwxyz -äö'
punctuation = r'!"#\$%&\'()\*\+,\./:;<=>\?@\[\\\]\^_`{\|}~'


# Poems -----------------------------------------------------------------------

def peruse_gutenberg():
    for dirpath, dirname, filenames in os.walk('gutenberg/gutenberg'):

        for f in filenames[1:]:
            filepath = dirpath + '/' + f
            poem, body = create_poem(f, filepath)
            curate_poem(poem, body)


def create_poem(filename, filepath):
    with open(filepath, 'r') as f:
        f = list(f)

    header, body = f[:4], f[4:]
    body = re.split(  # too many blank lines...
        r'\r\n',
        re.sub(r'[^A-Z]\r\n\r\n\r\n', '\r\n\r\n', ''.join(body)),
        )
    ebook_num = filename[2:-4]  # pg7000.txt > 7000
    header = [line.split(': ', 1)[1].strip('\r\n') for line in header]
    title, poet, released, updated = header
    poet = poet.split(' ')[-1]

    # create and save Poem object
    try:
        poem = finn.Poem(
            ebook_number=ebook_num,
            title=title,
            poet=poet,
            date_released=datetime.strptime(released, '%B %d, %Y'),
            )
    except ValueError:
        poem = finn.Poem(
            ebook_number=ebook_num,
            title=title,
            poet=poet,
            date_released=datetime.strptime(released, '%B, %Y'),
            last_updated=datetime.strptime(updated, '%B %d, %Y'),
            )

    finn.db.session.add(poem)
    finn.db.session.commit()

    return poem, body


def curate_poem(poem, body):
    tokenized_poem = []
    text = ''
    portion = 1

    for line in body:

        # if the line is a blank line, insert an HTML breakpoint
        if not line:
            text += '<br>'
            continue

        # if the line marks a NEW PAGE, create a new poem (this serves to split
        # each book of poems across several pages)
        if line == '<NEW PAGE>':
            tokenized_poem.append(text)
            portion += 1
            poem = duplicate_poem(poem, tokenized_poem, portion)
            tokenized_poem = []
            text = ''
            continue

        text += '<div>'

        # split line by punctuation, spaces, and newline characters:
        # 'päälle pään on taivosehen;' >
        # ['päälle', ' ', 'pään', ' ', 'on', ' ', 'taivosehen', ';']
        line = filter(None, re.split(
            r'(\r\n|[ ]+|[%s]|--)' % punctuation,
            line,
            ))

        for word in line:
            word = word.decode('utf-8', errors='replace')

            # if the word is a series of spaces, insert HTML non-breaking
            # spaces of an equivalent length
            if len(word) > 1 and word == len(word) * ' ':
                text += '&nbsp;' * len(word)
                continue

            # ignore any words that appear in all uppercase (e.g., acronyms)
            if word == word.upper():
                text += word
                continue

            word = word.lower()

            # find all u- and y-final diphthongs in word
            sequences = u_y_final_diphthongs(word.encode('utf-8'))
            sequences = filter(lambda seq: seq.group(1) is not None, sequences)

            # if the word contains any u- and y-final diphthongs and is only
            # composed of acceptable characters...
            if sequences and all(1 if i in characters else 0 for i in word):

                tokenized_poem.append(text)
                text = ''

                # find existing Token, or create a new Token object if one does
                # not already exist
                token = get_token(word)

                # create Variation object
                variation = finn.Variation(token=token.id, poem=poem.id)
                finn.db.session.add(variation)
                finn.db.session.commit()
                tokenized_poem.append(variation.id)

                # create Sequence objects
                curate_sequences(word, sequences, variation)

            else:
                text += word

        text += '</div>'

    tokenized_poem.append(text)
    poem.tokenized_poem = tokenized_poem
    finn.db.session.commit()

    print 'Poem!'


def duplicate_poem(poem, tokenized_poem, portion):
    poem.tokenized_poem = tokenized_poem

    new_poem = finn.Poem(
        ebook_number=poem.ebook_number,
        title=poem.title,
        poet=poem.poet,
        date_released=poem.date_released,
        last_updated=poem.last_updated,
        portion=portion,
        )

    finn.db.session.add(new_poem)
    finn.db.session.commit()

    print 'Semi-poem!'

    return new_poem


def u_y_final_diphthongs(word, strict=True):
    if strict:
        # this pattern searchs for VV sequences that ends in /u/ or /y/ that do
        # not appear within larger vowel sequences
        return list(re.finditer(
            r'(?<![ieäyöauo])(au|eu|ou|iu|iy|ey|äy|öy)(?:[^ieäyöauo]{1}|$)',
            # add |Au|Eu|Ou|Iu|Iy|Ey|Äy|Öy for manual search in Sublime
            word,
            ))

    # this pattern searchs for VV sequences that ends in /u/ or /y/, regardless
    # of their environments
    return list(re.finditer(
        r'(au|eu|ou|iu|iy|ey|äy|öy)',
        word,
        ))


def get_token(word):
    token = finn.find_token(word)

    if not token:
        token = finn.Token(orth=word)
        finn.db.session.add(token)

    token.is_gutenberg = True
    finn.db.session.commit()

    return token


def curate_sequences(word, sequences, variation):  # TOTES BROKEN
    previous = []

    for seq in sequences:
        i = seq.start(1)
        j = i + 2

        # eliminate duplicate matches
        if i not in previous:
            vv = seq.group(1).decode('utf-8')
            html = '%s<strong>%s</strong>%s' % (word[:i], vv, word[j:])

            # create Sequence object
            sequence = finn.Sequence(
                variation=variation.id,
                sequence=vv,
                html=html,
                )
            finn.db.session.add(sequence)

            previous.append(i)

    # commit sequences
    finn.db.session.commit()


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # peruse_gutenberg()
    pass
