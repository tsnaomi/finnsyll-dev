# coding=utf-8

from datetime import datetime
from finnsyll import db, Sequence, Token

from sqlalchemy import or_

# ADJUST() allows you to apply changes to all of the tokens in the database
# without completely rendering your computer incapable of doing anything else.


def ADJUST():
    '''Adjust tokens.'''
    print 'Adjusting tokens... ' + datetime.utcnow().strftime('%I:%M')

    count = Token.query.count()
    start = 0
    end = x = 1000

    while start + x < count:
        for token in Token.query.order_by(Token.id).slice(start, end):
            # APPLY ADJUSTMENTS HERE

            # -----------------------------------------------------------------
            pass

        db.session.commit()
        start = end
        end += x

    for token in Token.query.order_by(Token.id).slice(start, count):
        # APPLY ADJUSTMENTS HERE

        # ---------------------------------------------------------------------
        pass

    db.session.commit()

    print 'Adjustment complete. ' + datetime.utcnow().strftime('%I:%M')


def ADJUST_SEQUENCES():
    '''Adjust sequences.'''
    sequences = Sequence.query.filter(or_(
        Sequence.sequence.contains('ä'),
        Sequence.sequence.contains('ö'),
        ))

    for seq in sequences:
        vv = seq.sequence
        word = seq.v_sequence.t_variation.orth.lower()
        i = seq.html.find('<')
        j = i + 2
        seq.html = '%s<strong>%s</strong>%s' % (word[:i], vv, word[j:])

    db.session.commit()


if __name__ == '__main__':
    # ADJUST()
    ADJUST_SEQUENCES()
