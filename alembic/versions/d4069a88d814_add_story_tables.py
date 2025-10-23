"""add_story_tables

Revision ID: d4069a88d814
Revises: 56254216524f
Create Date: 2025-10-23 11:55:15.362539

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4069a88d814'
down_revision: Union[str, None] = '56254216524f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create stories table
    op.create_table(
        'stories',
        sa.Column('id', sa.String, primary_key=True),
        sa.Column('story_id', sa.String, nullable=False, unique=True, index=True),
        sa.Column('title', sa.String, nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('story_json', sa.JSON, nullable=False),  # Complete story structure
        sa.Column('scenes_json', sa.JSON, nullable=False),  # Processed scenes
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('organization_id', sa.String, nullable=False, index=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )
    
    # Create story_sessions table
    op.create_table(
        'story_sessions',
        sa.Column('id', sa.String, primary_key=True),
        sa.Column('session_id', sa.String, nullable=False, unique=True, index=True),
        sa.Column('user_id', sa.String, nullable=False, index=True),
        sa.Column('story_id', sa.String, nullable=False, index=True),
        sa.Column('status', sa.String, nullable=False, default='active'),  # active, paused, completed, archived
        sa.Column('state', sa.JSON, nullable=False),  # SessionState as JSON
        sa.Column('character_agents', sa.JSON, nullable=False),  # character_name -> agent_id mapping
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('organization_id', sa.String, nullable=False, index=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        
        # Foreign key to stories
        sa.ForeignKeyConstraint(['story_id'], ['stories.story_id'], ondelete='CASCADE'),
    )
    
    # Create indices for common queries
    op.create_index('idx_story_sessions_user_story', 'story_sessions', ['user_id', 'story_id'])
    op.create_index('idx_story_sessions_status', 'story_sessions', ['status'])


def downgrade() -> None:
    # Drop indices
    op.drop_index('idx_story_sessions_status')
    op.drop_index('idx_story_sessions_user_story')
    
    # Drop tables in reverse order
    op.drop_table('story_sessions')
    op.drop_table('stories')
