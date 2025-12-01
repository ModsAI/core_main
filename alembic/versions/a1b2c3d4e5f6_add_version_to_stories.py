"""add_version_to_stories

Revision ID: a1b2c3d4e5f6
Revises: 07ad6b9b1b82
Create Date: 2025-12-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '07ad6b9b1b82'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add version column to stories table for cache invalidation"""
    # Add version column - nullable to support stories without versions
    # Stories uploaded without version field will have NULL version
    op.add_column(
        'stories',
        sa.Column('version', sa.String(), nullable=True)
    )


def downgrade() -> None:
    """Remove version column from stories table"""
    op.drop_column('stories', 'version')

