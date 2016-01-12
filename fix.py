# coding=utf-8

import re

from finnsyll import db, Poem, replace_umlauts, Sequence
from gutenberg import u_y_final_diphthongs

# This file is set aside for fixing the many bugs that plague the poetry
# interface. Fu#$@&%*!


# Eliminate cases like pA<strong>Ay</strong>t (pAAyt) -------------------------


# Add Sequence objects for missed sequences -----------------------------------


# Add indications of missed sequences to poetry pages -------------------------

def mark_missing_sequences():
    for p in Poem.query.all():
        tp = list(p.tokenized_poem)

        for I, t in enumerate(tp):

            if isinstance(t, (str, unicode)):
                T = t.encode('utf-8')
                x = T
                matches = u_y_final_diphthongs(T)
                offset = 0

                for m in matches:
                    i = offset + m.start()
                    j = offset + m.end() - 1
                    insert = (
                        "<span style='font-size:30px;'><strong><span ",
                        "style='font-size:1px;'>@</span>%s</strong></span>",
                        ) % m.groups()[0]
                    x = T[:i] + insert + T[j:]
                    offset += 91
                    T = x

                tp[I] = T.decode('utf-8')

        p.tokenized_poem = tp

    db.session.commit()


# Fix umlaut bug in Sequence.html ---------------------------------------------

def fix_umlaut_bug():
    for s in Sequence.query.all():
        orth = s.v_sequence.t_variation.orth.lower()

        if s.html.replace('<strong>', '').replace('</strong>', '') != orth:
            orth = replace_umlauts(orth)
            html = replace_umlauts(s.html)
            vv = re.search(r'<strong>(.{2})</strong>', html).groups()[0]
            i = html.index('<')
            i = i - orth[:i].count('A') - orth[:i].count('O')
            i += 1 if vv[0] in 'AO' else 0
            good = orth[:i] + '<strong>' + vv + '</strong>' + orth[i + 2:]
            s.html = replace_umlauts(good, put_back=True)

    db.session.commit()


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # mark_missing_sequences()
    fix_umlaut_bug()
