# coding=utf-8

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
from flask.ext.migrate import Migrate, MigrateCommand
from flask.ext.seasurf import SeaSurf
from flask.ext.sqlalchemy import SQLAlchemy
from flask.ext.script import Manager
from flask.ext.bcrypt import Bcrypt
from functools import wraps
from math import ceil
from syllabifier.phonology import get_sonorities, get_weights
from syllabifier.v2 import syllabify

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
    applied_rules = db.Column(db.String(80, convert_unicode=True), default='')

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
            return self.syll.count('.') + 1

    @property
    def syllables(self):
        '''Return a list of the word's syllables.'''
        return self.test_syll.split('.')

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
            db.session.commit()

            return is_gold

        return False

    def syllabify(self):
        '''Algorithmically syllabify Token based on its orthography.'''
        # syllabifcations do not preserve capitalization
        token = self.orth.lower()
        self.test_syll, self.applied_rules = syllabify(token)

        if self.syll:
            self.update_gold()

    def correct(self, **kwargs):
        '''Save new attribute values to Token and update gold.'''
        for attr, value in kwargs.iteritems():
            if hasattr(self, attr):
                setattr(self, attr, value)

        # db.session.commit()  # TODO
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

    def render_html(self):
        '''Return text as an html string to be rendered on the frontend.

        This html string includes a modal for each word in the text. Each modal
        contains a form that will allow Arto to edit the word's Token, i.e.,
        Token.syll, Token.alt_syll1-3, and Token.is_compound.
        '''

        def gold_class(word):
            gold = word.is_gold
            return 'good' if gold else 'unverified' if gold is None else 'bad'

        html = u'<div class="doc-text">'

        for t in self.tokenized_text:

            if isinstance(t, int):
                word = Token.query.get(t)
                html += u' <a href="#modal"'

                html += (
                    u' onclick="populatemodal(\'%s\', \'%s\', \'%s\', \'%s\','
                    u' \'%s\', \'%s\', \'%s\', \'%s\', \'%s\', \'%s\');"') % (
                    word.id,
                    word.orth,
                    gold_class(word),
                    word.test_syll,
                    word.applied_rules,
                    word.syll or word.test_syll,
                    word.alt_syll1,
                    word.alt_syll2,
                    word.alt_syll3,
                    word.is_compound,
                    )

                html += u' class="word %s' % gold_class(word)

                if word.is_compound:
                    html += u' compound'

                if word.alt_syll1 or word.alt_syll2 or word.alt_syll3:
                    html += u' alt'

                html += u'"> %s </a>' % word.test_syll

            else:
                html += u'<span class="punct">%s</span>' % t

                if t == u'.':
                    html += u'<br><br>'

        if html.endswith(u'<br><br>'):
            html = html[:-8]  # fencepost

        html += u'</div>'

        return html

    def get_tokens(self):
        '''Return a list of the Tokens that appear in the text.'''
        tokens = []

        for ID in self.tokens:
            token = Token.query.get(ID)
            tokens.append(token)

        return tokens

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

        # if there are no unverified tokens but the document isn't marked as
        # reviewed, mark the document as reviewed; this would be the case if
        # all of the documents's tokens were verified in previous documents
        if unverified_count == 0:
            self.reviewed = True
            db.session.commit()


# Database functions ----------------------------------------------------------

def syllabify_tokens():
    '''Algorithmically syllabify all Tokens.

    This is done anytime a Token is instantiated. It *should* also be done
    anytime the syllabifying algorithm is updated.'''
    tokens = Token.query.all()

    for token in tokens:
        token.syllabify()

    db.session.commit()


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


def get_bad_tokens():
    '''Return all of the Tokens that are incorrectly syllabified.'''
    return Token.query.filter_by(is_gold=False).order_by(Token.lemma)


def get_good_tokens():
    '''Return all of the Tokens that are correctly syllabified.'''
    return Token.query.filter_by(is_gold=True).order_by(Token.lemma)


def get_unverified_tokens():
    '''Return Tokens with uncertain syllabifications.'''
    return Token.query.filter_by(is_gold=None).order_by(Token.lemma)


def get_unseen_lemmas():
    '''Return unseen lemmas with uncertain syllabfications.'''
    return Token.query.filter_by(freq=0).order_by(Token.lemma)


def get_unreviewed_documents():
    '''Return all unreviewed documents.'''
    docs = Document.query.filter_by(reviewed=False)
    docs = docs.order_by(Document.unique_count.desc()).limit(10)

    return docs


def get_numbers():
    '''Generate statistics.'''

    class Stats(object):
        _token_count = Token.query.count()
        _verified = Token.query.filter(Token.is_gold.isnot(None)).count()
        _gold = Token.query.filter_by(is_gold=True).count()
        _accuracy = (float(_gold) / _verified) * 100 if _gold else 0
        _remaining = _token_count - _verified
        _doc_count = Document.query.count()
        _reviewed = Document.query.filter_by(reviewed=True).count()

        token_count = format(_token_count, ',d')
        verified = format(_verified, ',d')
        gold = format(_gold, ',d')
        accuracy = round(_accuracy, 2)
        remaining = format(_remaining, ',d')
        doc_count = format(_doc_count, ',d')
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


def apply_form(http_form):
    # Apply changes to Token instance based on POST request
    try:
        orth = http_form['orth']
        syll = http_form['syll'] or http_form['test_syll']
        alt_syll1 = http_form['alt_syll1'] or ''
        alt_syll2 = http_form['alt_syll2'] or ''
        alt_syll3 = http_form['alt_syll3'] or ''
        is_compound = bool(http_form.getlist('is_compound'))
        is_stopword = bool(http_form.getlist('is_stopword'))
        token = Token.query.get(http_form['find'])

        token.correct(
            orth=orth,
            syll=syll,
            alt_syll1=alt_syll1,
            alt_syll2=alt_syll2,
            alt_syll3=alt_syll3,
            is_compound=is_compound,
            is_stopword=is_stopword,
            )

    except (AttributeError, KeyError, LookupError):
        pass


# Views -----------------------------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
@login_required
def main_view():
    '''List links to unverified texts (think: Table of Contents).'''
    stats = get_numbers()

    return render_template('main.html', stats=stats, kw='main')


@app.route('/doc/<id>', methods=['GET', 'POST'])
@login_required
def doc_view(id):
    '''Present detail view of specified doc, composed of editable Tokens.'''
    if request.method == 'POST':
        apply_form(request.form)

    doc = Document.query.get_or_404(id)
    TEXT = doc.render_html()

    return render_template('doc.html', doc=doc, TEXT=TEXT, kw='doc')


@app.route('/approve/approve/approve/doc/<id>', methods=['POST', ])
@login_required
def approve_doc_view(id):
    '''For all of the doc's unverified Tokens, set syll equal to test_syll.'''
    doc = Document.query.get_or_404(id)
    doc.verify_all_unverified_tokens()

    return redirect(url_for('doc_view', id=id))


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


@app.route('/lemma', defaults={'page': 1}, methods=['GET', 'POST'])
@app.route('/lemma/page/<int:page>')
def lemma_view(page):
    '''List all unverified unseen lemmas and process corrections.'''
    if request.method == 'POST':
        apply_form(request.form)

    tokens = get_unseen_lemmas()
    tokens, pagination = paginate(page, tokens)

    return render_template(
        'tokens.html',
        tokens=tokens,
        kw='lemma',
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

PER_PAGE = 40


class Pagination(object):

    def __init__(self, page, total_count):
        self.page = page
        self.per_page = PER_PAGE
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


def paginate(page, tokens):
    count = tokens.count()
    start = (page - 1) * PER_PAGE or 0
    end = min(start + PER_PAGE, count)

    try:
        tokens = tokens[start:end]

    except IndexError:
        if page != 1:
            abort(404)

    pagination = Pagination(page, count)

    return tokens, pagination


def url_for_other_page(page):
    args = request.view_args.copy()
    args['page'] = page
    return url_for(request.endpoint, **args)

app.jinja_env.globals['url_for_other_page'] = url_for_other_page


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    manager.run()
