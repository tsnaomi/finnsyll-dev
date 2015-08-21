"""empty message

Revision ID: 35775c59d61b
Revises: 38843212685d
Create Date: 2015-08-14 00:15:24.551870

"""

# revision identifiers, used by Alembic.
# revision = '35775c59d61b'
revision = '134b459dd021'
# down_revision = '1664a1daa8aa'
down_revision = '38843212685d'

from alembic import op
import sqlalchemy as sa


def upgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.add_column('Token', sa.Column('is_aamulehti', sa.Boolean(), nullable=True))
    op.add_column('Token', sa.Column('is_gutenberg', sa.Boolean(), nullable=True))
    op.create_table('Poem',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('title', sa.String(length=80, convert_unicode=True), nullable=True),
    sa.Column('poet', sa.Enum(u'J. H. Erkko', u'Aaro Hellaakoski', u'K\xf6ssi Kaatra', u'Uuno Kailas', u'V. A. Koskenniemi', u'Kaarlo Kramsu', u'Eino Leino', u'Elias L\xf6nnrot', u'Juhani Siljo', name='poet'), nullable=True),
    sa.Column('ebook_number', sa.Integer(), nullable=True),
    sa.Column('date_released', sa.DateTime(), nullable=True),
    sa.Column('last_updated', sa.DateTime(), nullable=True),
    sa.Column('tokenized_poem', sa.PickleType(), nullable=True),
    sa.Column('reviewed', sa.Boolean(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('Sequence',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('token_id', sa.Integer(), nullable=True),
    sa.Column('poem_id', sa.Integer(), nullable=True),
    sa.Column('sequence', sa.String(length=10, convert_unicode=True), nullable=True),
    sa.Column('html', sa.String(length=80, convert_unicode=True), nullable=True),
    sa.Column('split', sa.Enum('split', 'join', 'unknown', name='split'), nullable=True),
    sa.Column('scansion', sa.Enum('S', 'W', 'SW', 'WS', 'SS', 'WW', 'UNK', name='scansion'), nullable=True),
    sa.Column('note', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['poem_id'], ['Poem.id'], ),
    sa.ForeignKeyConstraint(['token_id'], ['Token.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    ### end Alembic commands ###


def downgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('Token', 'is_gutenberg')
    op.drop_column('Token', 'is_aamulehti')
    op.drop_table('Sequence')
    op.drop_table('Poem')
    ### end Alembic commands ###
