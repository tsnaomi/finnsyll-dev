# coding=utf-8

from datetime import datetime
from flask import (
    abort,
    flash,
    Flask,
    redirect,
    render_template,
    request,
    session,
    url_for,
    )
from flaskext.markdown import Markdown
from flask.ext.migrate import Migrate, MigrateCommand
from flask.ext.seasurf import SeaSurf
from flask.ext.sqlalchemy import SQLAlchemy
from flask.ext.script import Manager
from flask.ext.bcrypt import Bcrypt
from functools import wraps
from math import ceil
from syllabifier.phonology import get_sonorities, get_weights
from syllabifier.v4 import syllabify
from tabulate import tabulate
from werkzeug.exceptions import BadRequestKeyError

app = Flask(__name__, static_folder='_static', template_folder='_templates')
app.config.from_pyfile('finnsyll_config.py')

db = SQLAlchemy(app)
migrate = Migrate(app, db)
manager = Manager(app)
manager.add_command('db', MigrateCommand)

# To mirate database:
#     python finnsyll.py db init (only for initial migration)
#     python finnsyll.py db migrate
#     python finnsyll.py db upgrade

csrf = SeaSurf(app)
flask_bcrypt = Bcrypt(app)
markdown = Markdown(app)


# Models ----------------------------------------------------------------------

class Linguist(db.Model):
    __tablename__ = 'Linguist'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(40), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)

    def __init__(self, username, password):
        self.username = username
        self.password = flask_bcrypt.generate_password_hash(password)

    def __repr__(self):
        return self.username

    def __unicode__(self):
        return self.__repr__()


class Token(db.Model):
    __tablename__ = 'Token'
    id = db.Column(db.Integer, primary_key=True)

    # the word's orthography
    orth = db.Column(db.String(80, convert_unicode=True), nullable=False)

    # the word's lemma/citation form
    lemma = db.Column(db.String(80, convert_unicode=True), default='')

    # the syllabification that is estimated programmatically
    test_syll = db.Column(db.String(80, convert_unicode=True), default='')

    # a string of the rules applied in the test syllabfication
    rules = db.Column(db.String(80, convert_unicode=True), default='')

    # the correct syllabification (hand-verified)
    syll = db.Column(db.String(80, convert_unicode=True), default='')

    # an alternative syllabification (hand-verified)
    alt_syll1 = db.Column(db.String(80, convert_unicode=True), default='')

    # an alternative syllabification (hand-verified)
    alt_syll2 = db.Column(db.String(80, convert_unicode=True), default='')

    # an alternative syllabification (hand-verified)
    alt_syll3 = db.Column(db.String(80, convert_unicode=True), default='')

    # the word's part-of-speech
    pos = db.Column(db.String(80, convert_unicode=True), default='')

    # the word's morpho-syntactic description
    msd = db.Column(db.String(80, convert_unicode=True), default='')

    # the word's frequency in the Aamulehti-1999 corpus
    freq = db.Column(db.Integer, default=0)

    # a boolean indicating if the word is a compound
    is_compound = db.Column(db.Boolean, default=False)

    # a boolean indicating if the word is a stopword -- only if the
    # word's syllabification is lexically marked
    is_stopword = db.Column(db.Boolean, default=False)

    # a boolean indicating if the algorithm has estimated correctly
    is_gold = db.Column(db.Boolean, default=None)

    def __init__(self, orth, lemma, msd, pos, freq):
        self.orth = orth
        self.lemma = lemma
        self.msd = msd
        self.pos = pos
        self.freq = freq

    def __repr__(self):
        return self.orth

    def __unicode__(self):
        return self.__repr__()

    # Token attribute methods -------------------------------------------------

    @property
    def syllable_count(self):
        '''Return the number of syllables the word contains.'''
        if self.syll:
            return self.syll.count('.') + 1  # TODO

    @property
    def syllables(self):
        '''Return a list of the word's syllables.'''
        return self.test_syll.split('.')  # TODO

    @property
    def weights(self):
        '''Return the weight structure of the test syllabification.'''
        return get_weights(self.test_syll)

    @property
    def sonorities(self):
        '''Return the sonority structure of the test syllabification.'''
        return get_sonorities(self.test_syll)

    def is_lemma(self):
        '''Return True if the word is in its citation form, else False.'''
        return self.orth.lower() == self.lemma.replace('_', ' ').lower()

    # Syllabification methods -------------------------------------------------

    def update_gold(self):
        '''Compare test syllabifcation against true syllabification.

        Token.is_gold is True if the test syllabifcation matches the true
        syllabification. Otherwise, Token.is_fold is False.
        '''
        if self.test_syll and self.syll:
            is_gold = self.test_syll == self.syll

            if not is_gold:
                is_gold = self.test_syll == self.alt_syll1

            if not is_gold:
                is_gold = self.test_syll == self.alt_syll2

            if not is_gold:
                is_gold = self.test_syll == self.alt_syll3

            self.is_gold = is_gold

            return is_gold

        return False

    def syllabify(self):
        '''Algorithmically syllabify Token based on its orthography.'''
        # syllabifcations do not preserve capitalization
        token = self.orth.lower()
        self.test_syll, self.rules = syllabify(token)

        if self.syll:
            self.update_gold()

    def correct(self, **kwargs):
        '''Save new attribute values to Token and update gold.'''
        for attr, value in kwargs.iteritems():
            if hasattr(self, attr):
                setattr(self, attr, value)

        self.update_gold()


class Document(db.Model):
    __tablename__ = 'Document'
    id = db.Column(db.Integer, primary_key=True)

    # the name of the xml file in the Aamulehti-1999 corpus
    filename = db.Column(db.Text, unique=True)

    # a boolean indicating if all of the document's words have been reviewed
    reviewed = db.Column(db.Boolean, default=False)

    # the text as a tokenized list, incl. Token IDs and punctuation strings
    tokenized_text = db.Column(db.PickleType)

    # a list of IDs for each word as they appear in the text
    tokens = db.Column(db.PickleType)

    # number of unique Tokens that appear in the text
    unique_count = db.Column(db.Integer)

    def __init__(self, filename, tokenized_text, tokens):
        self.filename = filename
        self.tokenized_text = tokenized_text
        self.tokens = tokens
        self.unique_count = len(tokens)

    def __repr__(self):
        return self.filename

    def __unicode__(self):
        return self.__repr__()

    def query_document(self):
        '''Return a list of Tokens and puncts as they appear in the text.'''
        tokens = {t.id: t for t in self.get_tokens()}
        doc = [tokens.get(t, t) for t in self.tokenized_text]

        return doc

    def get_tokens(self):
        '''Return a list of the Tokens that appear in the text.'''
        return db.session.query(Token).filter(Token.id.in_(self.tokens)).all()

    def verify_all_unverified_tokens(self):
        '''For all of the text's unverified Tokens, set syll equal to test_syll.

        This function is intended for when all uverified Tokens have been
        correctly syllabified in test_syll. Proceed with caution.
        '''
        tokens = self.get_tokens()

        for token in tokens:
            if token.is_gold is None:
                token.correct(syll=token.test_syll)

        self.reviewed = True
        db.session.commit()

    def update_review(self):
        '''Set reviewed to True if all of the Tokens have been verified.'''
        tokens = self.get_tokens()
        unverified_count = 0

        for t in tokens:
            if t.is_gold is None:
                unverified_count += 1
                break

        # if there are no unverified tokens but the document isn't marked as
        # reviewed, mark the document as reviewed; this would be the case if
        # all of the documents's tokens were verified in previous documents
        if unverified_count == 0:
            self.reviewed = True


# Database functions ----------------------------------------------------------

def syllabify_tokens():
    '''Algorithmically syllabify all tokens.'''
    print 'Syllabifying...'

    count = Token.query.count()
    start = 0
    end = x = 1000

    while start + x < count:
        for token in Token.query.order_by(Token.id).slice(start, end):
            token.syllabify()

        db.session.commit()
        start = end
        end += x

    for token in Token.query.order_by(Token.id).slice(start, count):
        token.syllabify()

    db.session.commit()

    print 'Syllabifications complete.'


def transition(pdf=False):
    '''Syllabify tokens and create a transition report.'''
    compound = lambda t: 'C' if t.is_compound else None
    parse = lambda t: (t, [t.test_syll, t.rules])
    row = lambda t: ['>', t.test_syll, t.rules, compound(t)]

    tokens = Token.query.filter(Token.is_gold.isnot(None))

    Dict = lambda tokens=tokens: {
        'good': dict([parse(t) for t in tokens.filter_by(is_gold=True)]),
        'bad': dict([parse(t) for t in tokens.filter_by(is_gold=False)]),
        }

    PRE = Dict()

    for t in tokens:
        t.syllabify()

    POST = Dict()

    if PRE['good'] != POST['good']:
        good = set(PRE['bad'].keys()).intersection(POST['good'].keys())
        bad = set(PRE['good'].keys()).intersection(POST['bad'].keys())
        table1 = [PRE['bad'][t] + row(t) for t in good]
        table2 = [PRE['good'][t] + row(t) for t in bad]
        report = 'FROM BAD TO GOOD (%s)\n' % len(good)
        report += tabulate(table1)
        report += '\n\nFROM GOOD TO BAD (%s)\n' % len(bad)
        report += tabulate(table2)
        report += '\n\n%s BAD TOKENS' % len(POST['bad'])

        if pdf:
            filename = 'syllabifier/reports/%s.txt' % str(datetime.utcnow())

            with open(filename, 'w') as f:
                f.write(report.encode('utf-8'))

        print report

    else:
        'BAD TOKENS' % len(POST['bad'])

    db.session.rollback()


def find_token(orth):
    '''Retrieve token by its orthography.'''
    try:
        # ilike queries are case insensitive
        token = Token.query.filter(Token.orth.ilike(orth)).first()
        return token

    except KeyError:
        return None


def update_documents():
    '''Mark documents as reviewed if all of their tokens have been verified.'''
    docs = Document.query.filter_by(reviewed=False)

    for doc in docs:
        doc.update_review()

    db.seees.commit()


def get_bad_tokens():
    '''Return all of the tokens that are incorrectly syllabified.'''
    return Token.query.filter_by(is_gold=False).order_by(Token.is_compound)


def get_good_tokens():
    '''Return all of the tokens that are correctly syllabified.'''
    return Token.query.filter_by(is_gold=True).order_by(Token.lemma)


def get_unverified_tokens():
    '''Return tokens with uncertain syllabifications.'''
    return Token.query.filter_by(is_gold=None).order_by(Token.freq.desc())


def get_unseen_lemmas():
    '''Return unseen lemmas with uncertain syllabfications.'''
    return Token.query.filter_by(freq=0).order_by(Token.lemma)


def get_stopword_tokens():
    '''Return all unverified stopwords.'''
    tokens = Token.query.filter_by(is_stopword=True).filter_by(is_gold=None)
    tokens = tokens.order_by(Token.freq.desc())

    return tokens


def get_unreviewed_documents():
    '''Return all unreviewed documents.'''
    docs = Document.query.filter_by(reviewed=False)
    docs = docs.order_by(Document.unique_count).limit(10)

    return docs


def get_numbers():
    '''Generate statistics.'''

    class Stats(object):
        # _token_count = tokens.count()
        # _doc_count = Document.query.count()
        _token_count = 991730  # one less ping to the database
        _doc_count = 61529  # one less ping to the database

        # caculate accuracy excluding compounds
        _non_compound_verified = Token.query.filter(Token.is_gold.isnot(None)) \
            .filter_by(is_compound=False).count()
        _non_compound_gold = Token.query.filter_by(is_gold=True) \
            .filter_by(is_compound=False).count()
        _accuracy = (float(_non_compound_gold) / _non_compound_verified) * 100

        # calculate accuracy including compounds
        _verified = Token.query.filter(Token.is_gold.isnot(None)).count()
        _gold = Token.query.filter_by(is_gold=True).count()
        _compound_accuracy = (float(_gold) / _verified) * 100

        _remaining = _token_count - _verified
        _reviewed = Document.query.filter_by(reviewed=True).count()

        token_count = format(_token_count, ',d')
        doc_count = format(_doc_count, ',d')
        verified = format(_verified, ',d')
        gold = format(_gold, ',d')
        accuracy = round(_accuracy, 2)
        compound_accuracy = round(_compound_accuracy, 2)
        remaining = format(_remaining, ',d')
        reviewed = format(_reviewed, ',d')

    stats = Stats()

    return stats


# View helpers ----------------------------------------------------------------

@app.before_request
def renew_session():
    # Forgot why I did this... but I think it's important
    session.modified = True


def login_required(x):
    # View decorator requiring users to be authenticated to access the view
    @wraps(x)
    def decorator(*args, **kwargs):
        if session.get('current_user'):
            return x(*args, **kwargs)

        return redirect(url_for('login_view'))

    return decorator


@app.context_processor
def serve_docs():
    docs = get_unreviewed_documents()

    return dict(docs=docs)


def redirect_url(default='main_view'):
    # Redirect page to previous url or to main_view
    return request.referrer or url_for(default)


def apply_form(http_form, commit=True):
    # Apply changes to Token instance based on POST request
    try:
        token = Token.query.get(http_form['find'])
        orth = http_form.get('orth')
        syll = http_form['syll'] or http_form['test_syll']
        alt_syll1 = http_form.get('alt_syll1', '')
        alt_syll2 = http_form.get('alt_syll2', '')
        alt_syll3 = http_form.get('alt_syll3', '')

        try:
            is_compound = bool(http_form.getlist('is_compound'))
            is_stopword = bool(http_form.getlist('is_stopword'))

        except AttributeError:
            is_compound = bool(http_form.get('is_compound'))
            is_stopword = bool(http_form.get('is_stopword'))

        token.correct(
            orth=orth or token.orth,
            syll=syll,
            alt_syll1=alt_syll1,
            alt_syll2=alt_syll2,
            alt_syll3=alt_syll3,
            is_compound=is_compound,
            is_stopword=is_stopword,
            )

        if commit:
            db.session.commit()

    except (AttributeError, KeyError, LookupError):
        pass


# Views -----------------------------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
@login_required
def main_view():
    '''List links to unverified texts (think: Table of Contents).'''
    stats = get_numbers()

    return render_template('main.html', stats=stats, kw='main')


@app.route('/rules', methods=['GET', ])
@login_required
def rules_view():
    '''List syllabification rules.'''
    return render_template('rules.html', kw='rules')


@app.route('/doc/<id>', methods=['GET', 'POST'])
@login_required
def doc_view(id):
    '''Present detail view of specified doc, composed of editable Tokens.'''
    if request.method == 'POST':
        apply_form(request.form)

    doc = Document.query.get_or_404(id)
    TEXT = doc.query_document()

    scroll = request.form.get('scroll', None)

    return render_template(
        'doc.html',
        doc=doc,
        TEXT=TEXT,
        kw='doc',
        scroll=scroll,
        )


@app.route('/approve/approve/approve/doc/<id>', methods=['POST', ])
@login_required
def approve_doc_view(id):
    '''For all of the doc's unverified Tokens, set syll equal to test_syll.'''
    doc = Document.query.get_or_404(id)
    doc.verify_all_unverified_tokens()

    return redirect(url_for('doc_view', id=id))


@app.route('/find', methods=['GET', 'POST'])
@login_required
def find_view():
    '''Search for tokens by word and/or citation form.'''
    results, find = None, None

    if request.method == 'POST':

        if request.form.get('syll'):
            apply_form(request.form)

        find = request.form.get('search') or request.form['syll']
        FIND = find.strip().translate({ord('.'): None, })  # strip periods
        results = Token.query.filter(Token.orth.ilike(FIND))
        results = results.order_by(Token.is_gold)

    return render_template(
        'search.html',
        kw='find',
        results=results,
        find=find,
        )


@app.route('/contains', defaults={'page': 1}, methods=['GET', 'POST'])
@app.route('/contains/page/<int:page>')
@login_required
def contains_view(page):
    '''Search for tokens by word and/or citation form.'''
    results, find, count = None, None, None

    if request.method == 'POST':
        find = request.form.get('search')

        if request.form.get('syll'):
            apply_form(request.form)

        if '.' in find:
            results = Token.query.filter(Token.test_syll.contains(find))

        else:
            results = Token.query.filter(Token.orth.contains(find))

        count = results.count()

        try:
            results = results.order_by(Token.is_gold)[:500]

        except IndexError:
            results = results.order_by(Token.is_gold)

    return render_template(
        'search.html',
        kw='contains',
        results=results,
        find=find,
        count=count,
        )


@app.route('/bad', defaults={'page': 1}, methods=['GET', 'POST'])
@app.route('/bad/page/<int:page>')
def bad_view(page):
    '''List all incorrectly syllabified Tokens and process corrections.'''
    if request.method == 'POST':
        apply_form(request.form)

    tokens = get_bad_tokens()
    tokens, pagination = paginate(page, tokens)

    return render_template(
        'tokens.html',
        tokens=tokens,
        kw='bad',
        pagination=pagination,
        )


# TODO: apply bulk form
@app.route('/unverified', defaults={'page': 1}, methods=['GET', 'POST'])
@app.route('/unverified/page/<int:page>')
def unverified_view(page):
    '''List all unverified Tokens and process corrections.'''
    if request.method == 'POST':
        forms = {k: {} for k in range(1, 41)}
        attrs = ['find', 'syll', 'alt_syll1', 'alt_syl2', 'is_compound']

        for i in range(1, 41):
            for attr in attrs:

                try:
                    forms[i][attr] = request.form['%s_%s' % (attr, i)]

                except BadRequestKeyError:
                    pass

        for form in forms.itervalues():
            apply_form(form, commit=False)

        db.session.commit()

    tokens = get_unverified_tokens().slice(0, 400)
    tokens, pagination = paginate(page, tokens)

    return render_template(
        'tokens.html',
        tokens=tokens,
        kw='unverified',
        pagination=pagination,
        )


# TODO: apply bulk form
@app.route('/lemma', defaults={'page': 1}, methods=['GET', 'POST'])
@app.route('/lemma/page/<int:page>')
def lemma_view(page):
    '''List all unverified unseen lemmas and process corrections.'''
    if request.method == 'POST':
        forms = {k: {} for k in range(1, 41)}
        attrs = [
            'find',
            'syll',
            'alt_syll1',
            'alt_syll2',
            'alt_syll3',
            'is_compound',
            ]

        for i in range(1, 41):
            for attr in attrs:

                try:
                    forms[i][attr] = request.form['%s_%s' % (attr, i)]

                except BadRequestKeyError:
                    pass

        for form in forms.itervalues():
            apply_form(form, commit=False)

        db.session.commit()

    tokens = get_unseen_lemmas()
    tokens, pagination = paginate(page, tokens)

    return render_template(
        'tokens.html',
        tokens=tokens,
        kw='lemmas',
        pagination=pagination,
        )


# TODO: apply bulk form
@app.route('/stopword', defaults={'page': 1}, methods=['GET', 'POST'])
@app.route('/stopword/page/<int:page>')
def stopword_view(page):
    '''List all stopwords and process corrections.'''
    tokens = get_stopword_tokens()
    tokens, pagination = paginate(page, tokens)

    return render_template(
        'tokens.html',
        tokens=tokens,
        kw='stopwords',
        pagination=pagination,
        )


@app.route('/enter', methods=['GET', 'POST'])
def login_view():
    '''Sign in current user.'''
    if session.get('current_user'):
        return redirect(url_for('main_view'))

    if request.method == 'POST':
        username = request.form['username']
        linguist = Linguist.query.filter_by(username=username).first()

        if linguist is None or not flask_bcrypt.check_password_hash(
                linguist.password,
                request.form['password']
                ):
            flash('Invalid username and/or password.')

        else:
            session['current_user'] = linguist.username
            return redirect(url_for('main_view'))

    return render_template('enter.html')


@app.route('/leave')
def logout_view():
    '''Sign out current user.'''
    session.pop('current_user', None)

    return redirect(url_for('main_view'))


# Pagination ------------------------------------------------------------------

class Pagination(object):

    def __init__(self, page, per_page, total_count):
        self.page = page
        self.per_page = per_page
        self.total_count = total_count

    @property
    def pages(self):
        return int(ceil(self.total_count / float(self.per_page)))

    @property
    def has_prev(self):
        return self.page > 1

    @property
    def has_next(self):
        return self.page < self.pages

    def iter_pages(self):
        left_edge, left_current = 2, 2
        right_edge, right_current = 2, 5

        last = 0
        for num in xrange(1, self.pages + 1):
            if num <= left_edge or (
                num > self.page - left_current - 1 and
                num < self.page + right_current
                    ) or num > self.pages - right_edge:
                if last + 1 != num:
                    yield None
                yield num
                last = num


def paginate(page, tokens, per_page=40):
    count = tokens.count()
    start = (page - 1) * per_page or 0
    end = min(start + per_page, count)

    try:
        tokens = tokens[start:end]

    except IndexError:
        if page != 1:
            abort(404)

    pagination = Pagination(page, per_page, count)

    return tokens, pagination


def url_for_other_page(page):
    args = request.view_args.copy()
    args['page'] = page
    return url_for(request.endpoint, **args)

app.jinja_env.globals['url_for_other_page'] = url_for_other_page


# Jinja2 ----------------------------------------------------------------------

def goldclass(t):
    gold = t.is_gold
    return u'good' if gold else u'unverified' if gold is None else u'bad'

app.jinja_env.filters['goldclass'] = goldclass
app.jinja_env.tests['token'] = lambda t: hasattr(t, 'syll')


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    manager.run()
