"""merge story system and main migrations

Revision ID: 8c34935d6e7a
Revises: cce9a6174366, d4069a88d814
Create Date: 2025-10-27 18:44:03.609859

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8c34935d6e7a'
down_revision: Union[str, None] = ('cce9a6174366', 'd4069a88d814')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
