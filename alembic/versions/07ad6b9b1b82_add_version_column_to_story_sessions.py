"""add_version_column_to_story_sessions

Revision ID: 07ad6b9b1b82
Revises: 8c34935d6e7a
Create Date: 2025-11-05 21:45:51.138823

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '07ad6b9b1b82'
down_revision: Union[str, None] = '8c34935d6e7a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add version column to story_sessions table for optimistic locking"""
    # Add version column with default value of 1
    # nullable=True for backwards compatibility with existing rows
    op.add_column(
        'story_sessions',
        sa.Column('version', sa.Integer(), nullable=True, server_default='1')
    )
    
    # Backfill existing rows with version = 1
    op.execute("UPDATE story_sessions SET version = 1 WHERE version IS NULL")


def downgrade() -> None:
    """Remove version column from story_sessions table"""
    op.drop_column('story_sessions', 'version')
