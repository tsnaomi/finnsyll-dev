# coding=utf-8

from datetime import datetime
from app import db, Token
from random import sample

# ADJUST() allows you to apply changes to all of the tokens in the database
# without completely rendering your computer incapable of doing anything else.


def create_sets():
    '''Create training, development, and test sets.'''
    simplex = set(Token.query.filter_by(is_complex=False))  # 16967
    compounds = set(Token.query.filter_by(is_complex=True))  # 4033

    for label in ['dev', 'test']:
        s = sample(simplex, 1696)  # 10% of simplex gold set
        c = sample(compounds, 404)  # 10% of complex gold set
        simplex.difference_update(s)
        compounds.difference_update(c)

        for t in s + c:
            t.data = label

    for t in simplex.union(compounds):  # remaining 80% of gold set
        t.data = 'train'

    db.session.commit()


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


if __name__ == '__main__':
    ADJUST()
