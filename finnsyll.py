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
from sqlalchemy import or_
from syllabifier.compound import detect
from syllabifier.phonology import FOREIGN_FINAL, get_sonorities, get_weights
from syllabifier.v6 import syllabify
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

    # a string of the rules applied in test_syll1
    rules1 = db.Column(db.String(80, convert_unicode=True), default='')

    # a string of the rules applied in test_syll2
    rules2 = db.Column(db.String(80, convert_unicode=True), default='')

    # a string of the rules applied in test_syll3
    rules3 = db.Column(db.String(80, convert_unicode=True), default='')

    # a string of the rules applied in test_syll4
    rules4 = db.Column(db.String(80, convert_unicode=True), default='')

    # a string of the rules applied in test_syll5
    rules5 = db.Column(db.String(80, convert_unicode=True), default='')

    # a string of the rules applied in test_syll6
    rules6 = db.Column(db.String(80, convert_unicode=True), default='')

    # a string of the rules applied in test_syll7
    rules7 = db.Column(db.String(80, convert_unicode=True), default='')

    # a string of the rules applied in test_syll8
    rules8 = db.Column(db.String(80, convert_unicode=True), default='')

    # test syllabification
    test_syll1 = db.Column(db.String(80, convert_unicode=True), default='')

    # test syllabification
    test_syll2 = db.Column(db.String(80, convert_unicode=True), default='')

    # test syllabification
    test_syll3 = db.Column(db.String(80, convert_unicode=True), default='')

    # test syllabification
    test_syll4 = db.Column(db.String(80, convert_unicode=True), default='')

    # test syllabification
    test_syll5 = db.Column(db.String(80, convert_unicode=True), default='')

    # test syllabification
    test_syll6 = db.Column(db.String(80, convert_unicode=True), default='')

    # test syllabification
    test_syll7 = db.Column(db.String(80, convert_unicode=True), default='')

    # test syllabification
    test_syll8 = db.Column(db.String(80, convert_unicode=True), default='')

    # correct syllabification (hand-verified)
    syll1 = db.Column(db.String(80, convert_unicode=True), default='')

    # correct syllabification (hand-verified)
    syll2 = db.Column(db.String(80, convert_unicode=True), default='')

    # correct syllabification (hand-verified)
    syll3 = db.Column(db.String(80, convert_unicode=True), default='')

    # correct syllabification (hand-verified)
    syll4 = db.Column(db.String(80, convert_unicode=True), default='')

    # correct syllabification (hand-verified)
    syll5 = db.Column(db.String(80, convert_unicode=True), default='')

    # correct syllabification (hand-verified)
    syll6 = db.Column(db.String(80, convert_unicode=True), default='')

    # correct syllabification (hand-verified)
    syll7 = db.Column(db.String(80, convert_unicode=True), default='')

    # correct syllabification (hand-verified)
    syll8 = db.Column(db.String(80, convert_unicode=True), default='')

    # the word's part-of-speech
    pos = db.Column(db.String(80, convert_unicode=True), default='')

    # the word's morpho-syntactic description
    msd = db.Column(db.String(80, convert_unicode=True), default='')

    # the word's frequency in the Aamulehti-1999 corpus
    freq = db.Column(db.Integer, default=0)

    # a boolean indicating if the word is a compound
    is_compound = db.Column(db.Boolean, default=False)

    # a boolean indicating if the word is a non-delimited compound
    is_nondelimited_compound = db.Column(db.Boolean, default=False)

    # a boolean indicating if the syllabifier predicts the word is a compound
    is_test_compound = db.Column(db.Boolean, default=False)

    # a boolean indicating if the word is a stopword -- only if the word's
    # syllabification is lexically marked
    is_stopword = db.Column(db.Boolean, default=False)

    # a boolean indicating if the algorithm has estimated correctly
    is_gold = db.Column(db.Boolean, default=None)

    # a note field to jot down notes about the word
    note = db.Column(db.Text, default='')

    # a temporary boolean to indicate whether Arto had verified the token prior
    # to updating the database to accommodate variation in test syllabifcations
    verified = db.Column(db.Boolean, default=False)

    __mapper_args__ = {
        'order_by': [is_gold, is_compound, freq.desc()],
        }

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
    def syllable_count(self):  # TODO
        '''Return the number of syllables the word contains.'''
        return self.test_syll.count('.') + 1

        # This only takes into consideration the number of syllables in the
        # first syllabification. It also fails when counting the number of
        # syllables in delimited compounds.

    @property
    def syllables(self):  # TODO
        '''Return a list of the word's syllables.'''
        return self.test_syll.split('.')

        # This fails when listing the syllables of delimited compounds.

    @property
    def weights(self):  # TODO
        '''Return the weight structure of the test syllabification.'''
        return get_weights(self.test_syll)

        # This only takes into consiterdation the syllable weights of the
        # first syllabification.

    @property
    def sonorities(self):  # TODO
        '''Return the sonority structure of the test syllabification.'''
        return get_sonorities(self.test_syll)

        # This only takes into consiterdation the sonority structure of the
        # first syllabification.

    def readable_lemma(self):
        '''Return a rreadable form of the lemma.'''
        return self.lemma.replace('_', ' ')

    def is_lemma(self):
        '''Return True if the word is in its citation form, else False.'''
        return self.orth.lower() == self.readable_lemma.lower()

    # Syllabification methods -------------------------------------------------

    def test_sylls(self):
        ''' '''
        return set(filter(None, [
            self.test_syll1,
            self.test_syll2,
            self.test_syll3,
            self.test_syll4,
            self.test_syll5,
            self.test_syll6,
            self.test_syll7,
            self.test_syll8,
            ]))

    def sylls(self):
        ''' '''
        return set(filter(None, [
            self.syll1,
            self.syll2,
            self.syll3,
            self.syll4,
            self.syll5,
            self.syll6,
            self.syll7,
            self.syll8,
            ]))

    def syllabify(self):
        '''Programmatically syllabify Token based on its orthography.'''
        # syllabifcations do not preserve capitalization
        token = self.orth.lower()
        syllabifications = list(syllabify(token))

        for i, (test_syll, rules) in enumerate(syllabifications, start=1):
            setattr(self, 'test_syll' + str(i), test_syll)
            setattr(self, 'rules' + str(i), rules)

        if self.syll1:
            self.update_gold()

    def correct(self, **kwargs):
        '''Save new attribute values to Token and update gold status.'''
        for attr, value in kwargs.iteritems():
            if hasattr(self, attr):
                setattr(self, attr, value)

        self.update_gold()

    def update_gold(self):
        '''Compare test syllabifcations against true syllabifications.

        Token.is_gold is True iff all of the gold syllabifications are
        represented in the test syllabifications.
        '''
        self.is_gold = self.sylls().issubset(self.test_sylls())

    # Compound functions ------------------------------------------------------

    def detect_if_compound(self):
        '''Programmatically detect if the Token is a compound.'''
        self.is_test_compound = detect(self.orth.lower())

    # Evaluation methods ------------------------------------------------------

    @property
    def precision(self):
        ''' '''
        try:
            tests, sylls = self.test_sylls(), self.sylls()
            return round(len(tests.intersection(sylls)) * 1.0 / len(tests), 2)

        except ZeroDivisionError:
            return 0.0

    @property
    def recall(self):
        ''' '''
        try:
            tests, sylls = self.test_sylls(), self.sylls()
            return round(len(tests.intersection(sylls)) * 1.0 / len(sylls), 2)

        except ZeroDivisionError:
            return 0.0


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
    '''Syllabify all tokens.'''
    print 'Syllabifying... ' + datetime.utcnow().strftime('%I:%M')

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

    print 'Syllabifications complete. ' + datetime.utcnow().strftime('%I:%M')


def detect_compounds():
    '''Detect non-delimited compounds.'''
    print 'Detecting compounds... ' + datetime.utcnow().strftime('%I:%M')

    count = Token.query.count()
    start = 0
    end = x = 1000

    while start + x < count:
        for token in Token.query.order_by(Token.id).slice(start, end):
            token.detect_if_compound()

        db.session.commit()
        start = end
        end += x

    for token in Token.query.order_by(Token.id).slice(start, count):
        token.detect_if_compound()

    db.session.commit()

    print 'Detection complete. ' + datetime.utcnow().strftime('%I:%M')


def find_token(orth):
    '''Retrieve a token by its orthography.'''
    try:
        # ilike queries are case insensitive
        token = Token.query.filter(Token.orth.ilike(orth)).first()
        return token

    except KeyError:
        return None


# Baisc queries ---------------------------------------------------------------

def get_bad_tokens():
    '''Return all of the tokens that are incorrectly syllabified.'''
    return Token.query.filter_by(is_gold=False)


def get_good_tokens():
    '''Return all of the tokens that are correctly syllabified.'''
    return Token.query.filter_by(is_gold=True).order_by(Token.lemma)


def get_unverified_tokens():
    '''Return tokens with uncertain syllabifications.'''
    return Token.query.filter_by(is_gold=None)


def get_unseen_lemmas():
    '''Return unseen lemmas with uncertain syllabfications.'''
    return Token.query.filter_by(freq=0).order_by(Token.lemma)


def get_stopwords():
    '''Return all unverified stopwords.'''
    tokens = Token.query.filter_by(is_stopword=True)

    return tokens


def get_foreign_words():
    '''Return a list of potential foreign words and interjections.

    This function returns all of the words that do not end in either a vowel or
    coronal consonant.
    '''
    query = lambda c: Token.query.filter(Token.orth.endswith(c))
    tokens = [t for c in FOREIGN_FINAL for t in query(c)]
    tokens = sorted(tokens, key=lambda t: (t.is_gold, t.freq), reverse=True)

    return tokens


def get_notes():
    ''' '''
    return Token.query.filter(Token.note != '').order_by(Token.freq.desc())


# Variation queries -----------------------------------------------------------

def get_variation():
    '''Return tokens with alternative test or gold syllabifications.'''
    return Token.query.filter(or_(Token.syll2 != '', Token.test_syll2 != ''))


def get_unverified_variation():
    '''Return unverified tokens with alternative test syllabifications.'''
    tokens = Token.query.filter(Token.test_syll2 != '')
    tokens = tokens.filter(Token.is_gold.is_(None))

    return tokens


def get_test_verified_variation():
    '''Return verified tokens with only alternative test syllabifications.'''
    tokens = Token.query.filter_by(verified=True)
    tokens = tokens.filter(Token.test_syll2 != '').filter(Token.syll2 == '')

    return tokens


def get_gold_verified_variation():
    '''Return tokens with alternative syllabifications prior to migration.'''
    tokens = Token.query.filter_by(verified=True)
    tokens = tokens.filter(Token.syll2 != '').filter(Token.test_syll2 == '')

    return tokens


# Compound queries ------------------------------------------------------------

def get_test_compounds():
    '''Return tokens predicted to be compounds.'''
    return Token.query.filter_by(is_test_compound=True)


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


# @app.context_processor
def serve_docs():
    # Serve documents to navbar
    docs = Document.query.filter_by(reviewed=False)
    docs = docs.order_by(Document.unique_count).limit(10)

    return dict(docs=docs)


def apply_form(http_form, commit=True):
    # Apply changes to Token instance based on POST request
    try:
        token = Token.query.get(http_form['id'])
        syll1 = http_form['syll1']
        syll2 = http_form.get('syll2', '')
        syll3 = http_form.get('syll3', '')
        syll4 = http_form.get('syll4', '')
        # syll5 = http_form.get('syll5', '')
        # syll6 = http_form.get('syll6', '')
        # syll7 = http_form.get('syll7', '')
        # syll8 = http_form.get('syll8', '')
        note = http_form.get('note', '')

        try:
            is_compound = bool(http_form.getlist('is_compound'))
            # is_stopword = bool(http_form.getlist('is_stopword'))

        except AttributeError:
            is_compound = bool(http_form.get('is_compound'))
            # is_stopword = bool(http_form.get('is_stopword'))

        token.correct(
            syll1=syll1,
            syll2=syll2,
            syll3=syll3,
            syll4=syll4,
            # syll5=syll5,
            # syll6=syll6,
            # syll7=syll7,
            # syll8=syll8,
            is_compound=is_compound,
            # is_stopword=is_stopword,
            note=note,
            verified_again=True,
            )

        if commit:
            db.session.commit()

    except (AttributeError, KeyError, LookupError):
        pass


def apply_bulk_form(http_form):
    # Apply changes to multiple Token instances based on POST request
    forms = {k: {} for k in range(1, 41)}
    attrs = ['id', 'syll1', 'syll2', 'syll3', 'syll4', 'is_compound', 'note']

    for i in range(1, 41):
        for attr in attrs:
            try:
                forms[i][attr] = request.form['%s_%s' % (attr, i)]

            except BadRequestKeyError:
                pass

    for form in forms.itervalues():
        apply_form(form, commit=False)

    db.session.commit()


# Views -----------------------------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
@login_required
def main_view():
    '''List statistics on the syllabifier's performance.'''
    VERIFIED = Token.query.filter(Token.is_gold.isnot(None))
    GOLD = VERIFIED.filter_by(is_gold=True)

    token_count = 991730  # Token.query.count()
    doc_count = 61529  # Document.query.count()

    # caculate accuracy excluding compounds
    simplex_verified = VERIFIED.filter_by(is_compound=False).count()
    simplex_gold = GOLD.filter_by(is_compound=False).count()
    simplex_accuracy = (float(simplex_gold) / simplex_verified) * 100

    # calculate accuracy including compounds
    verified = VERIFIED.count()
    gold = GOLD.count()
    accuracy = (float(gold) / verified) * 100

    # calculate aamulehti numbers
    remaining = token_count - verified
    reviewed = 823  # Document.query.filter_by(reviewed=True).count()

    # calculate compound numbers
    compound_verified = verified - simplex_verified
    compound_gold = VERIFIED.filter_by(is_test_compound=True).count()
    compound_accuracy = (float(compound_gold) / compound_verified) * 100

    # calculate average precision and recall
    precision = float(sum([t.precision for t in VERIFIED])) / verified
    recall = float(sum([t.recall for t in VERIFIED])) / verified

    stats = {
        'token_count': format(token_count, ',d'),
        'doc_count': format(doc_count, ',d'),
        'verified': format(verified, ',d'),
        'gold': format(gold, ',d'),
        'simplex_accuracy': round(simplex_accuracy, 2),
        'accuracy': round(accuracy, 2),
        'remaining': format(remaining, ',d'),
        'compound_verified': format(compound_verified, ',d'),
        'compound_gold': format(compound_gold, ',d'),
        'compound_accuracy': round(compound_accuracy, 2),
        'reviewed': format(reviewed, ',d'),
        'precision': round(precision, 4),
        'recall': round(recall, 4)
        }

    return render_template('main.html', kw='main', stats=stats)


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


@app.route('/rules', methods=['GET', ])
@login_required
def rules_view():
    '''List syllabification rules.'''
    return render_template('rules.html', kw='rules')


@app.route('/notes/', defaults={'page': 1}, methods=['GET', 'POST'])
@app.route('/notes/page/<int:page>', methods=['GET', 'POST'])
def notes_view(page):
    '''List all tokens that contain notes.'''
    if request.method == 'POST':
        apply_form(request.form)

    tokens = get_notes()

    return render_template(
        'tokens.html',
        tokens=tokens,
        kw='notes',
        )


@app.route('/contains', methods=['GET', 'POST'])
@login_required
def contains_view():
    '''Search for tokens by word and/or citation form.'''
    results, find, count = None, None, None

    if request.method == 'POST':
        find = request.form.get('search')

        if request.form.get('syll1'):
            apply_form(request.form)

        if '.' in find:
            results = Token.query.filter(or_(
                Token.test_syll1.contains(find),
                Token.test_syll2.contains(find),
                Token.test_syll3.contains(find),
                Token.test_syll4.contains(find),
                ))

        else:
            results = Token.query.filter(Token.orth.contains(find))

        count = format(results.count(), ',d')

        try:
            results = results[:500]

        except IndexError:
            pass

    return render_template(
        'search.html',
        kw='contains',
        results=results,
        find=find,
        count=count,
        )


@app.route('/find', methods=['GET', 'POST'])
@login_required
def find_view():
    '''Search for tokens by word and/or citation form.'''
    results, find = None, None

    if request.method == 'POST':

        if request.form.get('syll1'):
            apply_form(request.form)

        find = request.form.get('search') or request.form['syll1']
        FIND = find.strip().translate({ord('.'): None, })  # strip periods
        # FIND = find.strip().translate(None, '.')  # strip periods
        results = Token.query.filter(Token.orth.ilike(FIND))
        results = results if results.count() > 0 else None

    return render_template(
        'search.html',
        kw='find',
        results=results,
        find=find,
        )


@app.route('/unverified', defaults={'page': 1}, methods=['GET', 'POST'])
@app.route('/unverified/page/<int:page>', methods=['GET', 'POST'])
def unverified_view(page):
    '''List all unverified Tokens and process corrections.'''
    if request.method == 'POST':
        apply_bulk_form(request.form)

    tokens = get_unverified_tokens().slice(0, 200)
    tokens, pagination = paginate(page, tokens, per_page=10)

    return render_template(
        'tokens.html',
        tokens=tokens,
        kw='unverified',
        pagination=pagination,
        )


@app.route('/bad', defaults={'page': 1}, methods=['GET', 'POST'])
@app.route('/bad/page/<int:page>', methods=['GET', 'POST'])
def bad_view(page):
    '''List all incorrectly syllabified Tokens and process corrections.'''
    if request.method == 'POST':
        apply_form(request.form)

    tokens = get_bad_tokens()
    count = format(tokens.count(), ',d')
    tokens, pagination = paginate(page, tokens)

    return render_template(
        'tokens.html',
        tokens=tokens,
        kw='bad',
        pagination=pagination,
        count=count,
        )


@app.route('/lemma', defaults={'page': 1}, methods=['GET', 'POST'])
@app.route('/lemma/page/<int:page>', methods=['GET', 'POST'])
def lemma_view(page):
    '''List all unverified unseen lemmas and process corrections.'''
    if request.method == 'POST':
        apply_bulk_form(request.form)

    tokens = get_unseen_lemmas()
    tokens, pagination = paginate(page, tokens)

    return render_template(
        'tokens.html',
        tokens=tokens,
        kw='lemmas',
        pagination=pagination,
        )


@app.route('/variation/', defaults={'page': 1}, methods=['GET', 'POST'])
@app.route('/variation/page/<int:page>', methods=['GET', 'POST'])
def variation_view(page):
    '''List all ambiguous tokens and process corrections.'''
    if request.method == 'POST':
        apply_form(request.form)

    tokens = get_variation()
    count = format(tokens.count(), ',d')
    tokens, pagination = paginate(page, tokens)

    return render_template(
        'tokens.html',
        tokens=tokens,
        kw='variation',
        pagination=pagination,
        count=count,
        )


@app.route('/<query>', defaults={'page': 1}, methods=['GET', 'POST'])
@app.route('/<query>/page/<int:page>', methods=['GET', 'POST'])
def hidden_view(page, query):
    '''List special queries.'''
    if request.method == 'POST':
        apply_form(request.form)

    # Monosyllabic test syllabifications
    mono = lambda: Token.query.filter(~Token.test_syll1.contains('.'))

    # Test compounds
    compounds = lambda: get_test_compounds()

    # Tokens with four+ test syllabifications
    four = lambda: Token.query.filter(Token.test_syll4 != '')

    queries = {
        'mono': mono,
        'compounds': compounds,
        'four': four,
        }

    try:
        tokens = queries[query]()
        tokens, pagination = paginate(page, tokens)

    except KeyError:
        abort(404)

    return render_template(
        'tokens.html',
        tokens=tokens,
        kw='hidden',
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


# Jinja2 ----------------------------------------------------------------------

def goldclass(t):
    gold = t.is_gold
    gold = u'good' if gold else u'unverified' if gold is None else u'bad'
    compound = ' compound' if t.is_compound else ''

    return gold + compound


def js_safe(s):
    s = s.replace('\r\n', '&#13;&#10;')
    s = s.replace('(', '&#40;').replace(')', '&#41;')
    s = s.replace('"', '&#34;').replace("'", '&#34;')

    return s

app.jinja_env.filters['goldclass'] = goldclass
app.jinja_env.filters['js_safe'] = js_safe
app.jinja_env.tests['token'] = lambda t: hasattr(t, 'syll1')


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


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    manager.run()
