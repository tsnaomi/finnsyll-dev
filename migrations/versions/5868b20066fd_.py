"""empty message

Revision ID: 5868b20066fd
Revises: 1cc7d716cf87
Create Date: 2015-08-20 23:13:34.693599

"""

# revision identifiers, used by Alembic.
revision = '5868b20066fd'
down_revision = '1cc7d716cf87'

from alembic import op
import sqlalchemy as sa


def upgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.add_column('Sequence', sa.Column('is_odd', sa.Boolean(), nullable=True))
    ### end Alembic commands ###


def downgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('Sequence', 'is_odd')
    ### end Alembic commands ###