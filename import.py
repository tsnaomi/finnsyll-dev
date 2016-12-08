# coding=utf-8

import pickle
import re

from collections import namedtuple
from sqlalchemy import or_

try:
    from app import Poem, Variation, Sequence, db
    Poet, Book, Section, VV = None, None, None, None

except ImportError:
    from app import Poet, Book, VV, db
    Poem, Variation, Sequence = None, None, None


Seq = namedtuple('Seq', 'seq index split scansion note line orth poet title')


# old database (finnsyll_backup_12-06-2016) -----------------------------------

def pickle_annotations():
    '''Pickle annotations from the old database.'''
    annotations = []

    # query all sequences containing any annotations
    sequences = Sequence.query.filter(or_(
        Sequence.split.isnot(None),
        Sequence.scansion.isnot(None),
        Sequence.note != '',
        ))

    # gather the annotations
    for seq in sequences:
        annotations.append(_get_Seq(seq))

    # pickle the annotations
    with open('annotations.pickle', 'w') as f:
        pickle.dump(annotations, f, pickle.HIGHEST_PROTOCOL)


def _get_Seq(seq):
    '''Return an annotation object.'''
    return Seq(
        seq=seq.sequence,
        index=seq.html.find('<'),
        split=seq.split,
        scansion='unknown' if seq.scansion == 'UNK' else seq.scansion,
        note=seq.note,
        line=_get_line(seq),
        orth=seq.orth,
        poet=seq.v_sequence.p_variation.poet,  # check match
        title=seq.v_sequence.p_variation.title,  # check match
        )


def _get_line(seq):
    '''Mimic VV.line.'''
    text = seq.v_sequence.p_variation.tokenized_poem
    index = text.index(seq.v_sequence.id)
    try:
        pre = re.split(r'\n|</div>|<div>|<br>', text[index - 1])[-1]
    except IndexError:
        pre = ''
    try:
        post = re.split(r'\n|</div>|<div>|<br>', text[index + 1])[0]
    except IndexError:
        post = ''
    line = '%s%s%s' % (pre, seq.orth.upper(), post)
    line = line.replace('&nbsp;', ' ')

    return line


# new database ----------------------------------------------------------------

def import_annotations():
    '''Import the pickled annoatations.'''
    # load the annotations
    with open('annotations.pickle', 'r+') as f:
        annotations = pickle.load(f)

    for seq in annotations:

        # filter VV sequences by poet, book title, sequence, and index
        poet = Poet.query.filter_by(surname=seq.poet).one()
        book = Book.query.filter_by(title=seq.title, poet_id=poet.id).one()
        VVs = VV.query.filter_by(
            book_id=book.id,
            sequence=seq.seq,
            index=seq.index,
            scansion=None,
            note='',
            ).all()

        # filter VV sequences by orth and line
        VVs = filter(lambda x: seq.orth == x.orth and x.line in seq.line, VVs)

        # store old annotations
        if len(VVs) > 0:
            vv = VVs[0]
            vv.correct(split=seq.split, scansion=seq.scansion, note=seq.note)
            db.session.commit()

        else:
            # print seq.orth, seq.index, seq.seq, seq.title, seq.poet
            # print '\t', seq.line
            # print '\t', seq.split, seq.scansion
            # ...
            # auringon 0 au Runoja Siljo
            #     AURINGON,
            #     join S
            continue

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # pickle_annotations()
    import_annotations()
    pass
