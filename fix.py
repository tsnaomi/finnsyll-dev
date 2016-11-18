# coding=utf-8

import csv

from app import db, Token, update_performance

with open('fix.csv', 'rb') as f:
    table = csv.reader(f, delimiter=',')
    table = [row for row in table][2:]

for id, _, _, note, _, _, _, _, s1, s2, _, _, fixed, _, _ in table:
    t = Token.query.get(int(id))
    t.note = note.decode('utf-8')

    if fixed:
        t.syll1 = s1.decode('utf-8')
        t.syll2 = s2.decode('utf-8')

    t.update_gold()

db.session.commit()
update_performance()
