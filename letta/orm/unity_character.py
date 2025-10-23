"""
Unity Character ORM Model

This model stores Unity character metadata and links them to Letta agents.
Follows the same patterns as existing Letta ORM models (Agent, Block, etc.)
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, Boolean, DateTime, Index, String, Text
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import Mapped, mapped_column

from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase

if TYPE_CHECKING:
    from letta.orm.agent import Agent
    from letta.orm.user import User


class UnityCharacter(SqlalchemyBase, OrganizationMixin, AsyncAttrs):
    """
    Unity Character Model - Links Unity characters to Letta agents
    
    This model stores the metadata for characters that Unity creates,
    maintaining the link between Unity's character ID and the underlying
    Letta agent that powers the character's intelligence.
    """
    
    __tablename__ = "unity_characters"
    __table_args__ = (
        # Index for fast lookups by Unity character ID
        Index("ix_unity_characters_unity_id", "unity_character_id"),
        # Index for fast lookups by Letta agent ID  
        Index("ix_unity_characters_agent_id", "letta_agent_id"),
        # Index for created_at queries
        Index("ix_unity_characters_created_at", "created_at", "id"),
        # Index for game_id lookups (multiple characters per game)
        Index("ix_unity_characters_game_id", "game_id"),
    )

    # Primary key - auto-generated UUID
    id: Mapped[str] = mapped_column(
        String, 
        primary_key=True, 
        default=lambda: f"unity-char-{uuid.uuid4()}",
        doc="Internal unique identifier for this Unity character record"
    )

    # Unity Integration Fields
    unity_character_id: Mapped[str] = mapped_column(
        String, 
        unique=True, 
        nullable=False,
        doc="The character ID that Unity uses to reference this character"
    )
    
    letta_agent_id: Mapped[str] = mapped_column(
        String, 
        nullable=False,
        doc="The Letta agent ID that powers this character's intelligence"
    )

    # Character Profile Fields
    character_name: Mapped[str] = mapped_column(
        String, 
        nullable=False,
        doc="Human-readable name of the character (e.g., 'Shopkeeper Bob')"
    )
    
    personality: Mapped[str] = mapped_column(
        Text, 
        nullable=False,
        doc="Character personality description for agent initialization"
    )
    
    backstory: Mapped[str] = mapped_column(
        Text, 
        nullable=False,
        doc="Character backstory and history for context"
    )
    
    voice_style: Mapped[Optional[str]] = mapped_column(
        String, 
        nullable=True,
        doc="Voice/speaking style description (e.g., 'cheerful and talkative')"
    )
    
    role: Mapped[str] = mapped_column(
        String, 
        nullable=False,
        doc="Character role in the story (e.g., 'merchant_npc', 'main_character')"
    )
    
    location: Mapped[Optional[str]] = mapped_column(
        String, 
        nullable=True,
        doc="Default location where this character appears"
    )

    # Game Context
    game_id: Mapped[str] = mapped_column(
        String, 
        nullable=False,
        doc="Identifier for the game/story this character belongs to"
    )

    # Character Configuration
    character_tier: Mapped[str] = mapped_column(
        String, 
        nullable=False, 
        default="dedicated",
        doc="Character tier: 'dedicated' (own agent), 'shared' (shared agent pool), 'template' (static responses)"
    )
    
    is_active: Mapped[bool] = mapped_column(
        Boolean, 
        nullable=False, 
        default=True,
        doc="Whether this character is currently active in the game"
    )

    # Metadata and Configuration
    character_config: Mapped[Optional[dict]] = mapped_column(
        JSON, 
        nullable=True,
        doc="Additional character configuration (animations, special behaviors, etc.)"
    )
    
    usage_stats: Mapped[Optional[dict]] = mapped_column(
        JSON, 
        nullable=True,
        doc="Usage statistics (interaction count, last used, etc.)"
    )

    # Audit Fields (inherited pattern from other models)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow,
        doc="When this character was registered"
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow,
        doc="When this character was last updated"
    )

    def __repr__(self) -> str:
        return (
            f"UnityCharacter("
            f"id='{self.id}', "
            f"unity_character_id='{self.unity_character_id}', "
            f"character_name='{self.character_name}', "
            f"role='{self.role}', "
            f"character_tier='{self.character_tier}', "
            f"is_active={self.is_active}"
            f")"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "unity_character_id": self.unity_character_id,
            "letta_agent_id": self.letta_agent_id,
            "character_name": self.character_name,
            "personality": self.personality,
            "backstory": self.backstory,
            "voice_style": self.voice_style,
            "role": self.role,
            "location": self.location,
            "game_id": self.game_id,
            "character_tier": self.character_tier,
            "is_active": self.is_active,
            "character_config": self.character_config,
            "usage_stats": self.usage_stats,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
