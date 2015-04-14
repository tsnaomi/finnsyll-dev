"""empty message

Revision ID: d7105ab189
Revises: None
Create Date: 2015-04-14 15:52:34.477645

"""

# revision identifiers, used by Alembic.
revision = 'd7105ab189'
down_revision = None

from alembic import op
import sqlalchemy as sa


def upgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.create_table('Token',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('orth', sa.String(length=80, convert_unicode=True), nullable=False),
    sa.Column('lemma', sa.String(length=80, convert_unicode=True), nullable=True),
    sa.Column('test_syll', sa.String(length=80, convert_unicode=True), nullable=True),
    sa.Column('applied_rules', sa.String(length=80, convert_unicode=True), nullable=True),
    sa.Column('syll', sa.String(length=80, convert_unicode=True), nullable=True),
    sa.Column('alt_syll1', sa.String(length=80, convert_unicode=True), nullable=True),
    sa.Column('alt_syll2', sa.String(length=80, convert_unicode=True), nullable=True),
    sa.Column('alt_syll3', sa.String(length=80, convert_unicode=True), nullable=True),
    sa.Column('pos', sa.String(length=80, convert_unicode=True), nullable=True),
    sa.Column('msd', sa.String(length=80, convert_unicode=True), nullable=True),
    sa.Column('freq', sa.Integer(), nullable=True),
    sa.Column('is_compound', sa.Boolean(), nullable=True),
    sa.Column('is_stopword', sa.Boolean(), nullable=True),
    sa.Column('is_gold', sa.Boolean(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('Document',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('filename', sa.Text(), nullable=True),
    sa.Column('reviewed', sa.Boolean(), nullable=True),
    sa.Column('tokenized_text', sa.PickleType(), nullable=True),
    sa.Column('tokens', sa.PickleType(), nullable=True),
    sa.Column('unique_count', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('filename')
    )
    op.create_table('Linguist',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=40), nullable=False),
    sa.Column('password', sa.String(length=80), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('username')
    )
    ### end Alembic commands ###


def downgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('Linguist')
    op.drop_table('Document')
    op.drop_table('Token')
    ### end Alembic commands ###
