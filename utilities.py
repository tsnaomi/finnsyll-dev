# coding=utf-8


def encode(u):
    '''Replace umlauts and convert "u" to a byte string.'''
    return u.replace(u'ä', u'{').replace(u'ö', u'|').replace(u'Ä', u'{') \
        .replace(u'Ö', u'|').encode('utf-8')
