"""
Story ORM Models

Database models for story management and session tracking.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.orm.sqlalchemy_base import SqlalchemyBase


class Story(SqlalchemyBase):
    """Story definition and metadata"""

    __tablename__ = "stories"
    __table_args__ = (
        Index("ix_stories_story_id", "story_id"),
        Index("ix_stories_organization_id", "organization_id"),
    )

    # Primary key
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: f"story-{uuid.uuid4()}")

    # Story identification
    story_id: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Story content (stored as JSON)
    story_json: Mapped[dict] = mapped_column(JSON, nullable=False)  # Original story structure
    scenes_json: Mapped[dict] = mapped_column(JSON, nullable=False)  # Processed scenes

    # Metadata (using column name 'metadata_' to avoid SQLAlchemy reserved name)
    story_metadata: Mapped[Optional[dict]] = mapped_column("metadata_", JSON, nullable=True)

    # Multi-tenancy
    organization_id: Mapped[str] = mapped_column(String, nullable=False, index=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    sessions: Mapped[list["StorySession"]] = relationship("StorySession", back_populates="story", cascade="all, delete-orphan")


class StorySession(SqlalchemyBase):
    """A user's playthrough of a story"""

    __tablename__ = "story_sessions"
    __table_args__ = (
        Index("ix_story_sessions_session_id", "session_id"),
        Index("ix_story_sessions_user_id", "user_id"),
        Index("ix_story_sessions_story_id", "story_id"),
        Index("idx_story_sessions_user_story", "user_id", "story_id"),
        Index("idx_story_sessions_status", "status"),
        Index("ix_story_sessions_organization_id", "organization_id"),
    )

    # Primary key
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: f"session-{uuid.uuid4()}")

    # Session identification
    session_id: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    story_id: Mapped[str] = mapped_column(String, ForeignKey("stories.story_id", ondelete="CASCADE"), nullable=False, index=True)

    # Session status
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")  # active, paused, completed, archived

    # Session state (stored as JSON)
    state: Mapped[dict] = mapped_column(JSON, nullable=False)  # SessionState schema

    # Optimistic locking version (incremented on each update to prevent race conditions)
    # Nullable=True for backwards compatibility with existing databases
    version: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=1, server_default="1")

    # Character-to-agent mappings
    character_agents: Mapped[dict] = mapped_column(JSON, nullable=False)  # character_name -> agent_id

    # Metadata (using column name 'metadata_' to avoid SQLAlchemy reserved name)
    session_metadata: Mapped[Optional[dict]] = mapped_column("metadata_", JSON, nullable=True)

    # Multi-tenancy
    organization_id: Mapped[str] = mapped_column(String, nullable=False, index=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    story: Mapped["Story"] = relationship("Story", back_populates="sessions")

