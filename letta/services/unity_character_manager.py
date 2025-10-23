"""
Unity Character Manager Service

Handles business logic for Unity character registration, management, and Letta agent integration.
Follows the same patterns as AgentManager, BlockManager, etc.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import sqlalchemy as sa
from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.exc import IntegrityError

from letta.constants import LETTA_TOOL_SET
from letta.log import get_logger
from letta.orm.unity_character import UnityCharacter
from letta.orm.errors import NoResultFound, UniqueConstraintViolationError
from letta.otel.tracing import trace_method
from letta.schemas.agent import AgentType, CreateAgent
from letta.schemas.unity_character import (
    CharacterTier,
    UnityCharacter as UnityCharacterSchema,
    UnityCharacterCreate,
    UnityCharacterUpdate,
)
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry

logger = get_logger(__name__)


class UnityCharacterManager:
    """Manager class to handle business logic related to Unity Characters."""

    def __init__(self):
        """Initialize the Unity Character Manager"""
        self.db = db_registry.get_db()
        logger.info("🎭 Unity Character Manager initialized")

    @trace_method
    async def create_unity_character(
        self,
        character_request: UnityCharacterCreate,
        actor: PydanticUser,
    ) -> UnityCharacterSchema:
        """
        Create a new Unity character and its associated Letta agent.
        
        This method:
        1. Validates the character doesn't already exist
        2. Creates a Letta agent with the character's personality
        3. Creates and stores the Unity character record
        4. Returns the complete character information
        
        Args:
            character_request: Character creation request
            actor: The user creating the character
            
        Returns:
            The created Unity character
            
        Raises:
            UniqueConstraintViolationError: If unity_character_id already exists
            ValueError: If character validation fails
        """
        logger.info(
            f"🎨 Creating Unity character: {character_request.character_name} "
            f"(ID: {character_request.unity_character_id}, Game: {character_request.game_id})"
        )
        
        try:
            # Check if character ID already exists
            async with self.db.get_async_session() as session:
                existing_check = select(UnityCharacter).where(
                    UnityCharacter.unity_character_id == character_request.unity_character_id
                )
                result = await session.execute(existing_check)
                existing_character = result.scalar_one_or_none()
                
                if existing_character:
                    logger.error(f"❌ Unity character ID '{character_request.unity_character_id}' already exists")
                    raise UniqueConstraintViolationError(
                        f"Unity character ID '{character_request.unity_character_id}' already exists"
                    )

                # Step 1: Create the Letta agent for this character
                letta_agent_id = await self._create_letta_agent_for_character(
                    character_request, actor
                )
                logger.info(f"✅ Created Letta agent {letta_agent_id} for character {character_request.character_name}")

                # Step 2: Create the Unity character record
                unity_character = UnityCharacter(
                    unity_character_id=character_request.unity_character_id,
                    letta_agent_id=letta_agent_id,
                    character_name=character_request.character_name,
                    personality=character_request.personality,
                    backstory=character_request.backstory,
                    voice_style=character_request.voice_style,
                    role=character_request.role,
                    location=character_request.location,
                    game_id=character_request.game_id,
                    character_tier=character_request.character_tier.value,
                    is_active=True,
                    character_config=character_request.character_config or {},
                    usage_stats={
                        "interaction_count": 0,
                        "created_timestamp": datetime.utcnow().isoformat(),
                        "last_interaction": None,
                    },
                    organization_id=actor.organization_id,
                )

                # Add and commit to database
                session.add(unity_character)
                await session.commit()
                await session.refresh(unity_character)

                logger.info(
                    f"✅ Successfully created Unity character '{character_request.character_name}' "
                    f"with agent {letta_agent_id}"
                )

                # Convert to schema and return
                return self._convert_to_schema(unity_character)

        except IntegrityError as e:
            logger.error(f"❌ Database integrity error creating Unity character: {e}")
            # Clean up the Letta agent if we created it
            if 'letta_agent_id' in locals():
                await self._cleanup_letta_agent(letta_agent_id, actor)
            raise UniqueConstraintViolationError(f"Unity character creation failed: {e}")
        
        except Exception as e:
            logger.error(f"❌ Unexpected error creating Unity character: {e}", exc_info=True)
            # Clean up the Letta agent if we created it
            if 'letta_agent_id' in locals():
                await self._cleanup_letta_agent(letta_agent_id, actor)
            raise

    @trace_method
    async def get_unity_character_by_id(
        self,
        unity_character_id: str,
        actor: PydanticUser,
    ) -> Optional[UnityCharacterSchema]:
        """
        Get Unity character by Unity character ID.
        
        Args:
            unity_character_id: The Unity character ID
            actor: The requesting user
            
        Returns:
            Unity character if found, None otherwise
        """
        logger.debug(f"🔍 Getting Unity character by ID: {unity_character_id}")
        
        async with self.db.get_async_session() as session:
            query = select(UnityCharacter).where(
                and_(
                    UnityCharacter.unity_character_id == unity_character_id,
                    UnityCharacter.organization_id == actor.organization_id,
                )
            )
            result = await session.execute(query)
            unity_character = result.scalar_one_or_none()
            
            if unity_character:
                logger.debug(f"✅ Found Unity character: {unity_character.character_name}")
                return self._convert_to_schema(unity_character)
            else:
                logger.debug(f"❌ Unity character not found: {unity_character_id}")
                return None

    @trace_method
    async def list_unity_characters(
        self,
        actor: PydanticUser,
        game_id: Optional[str] = None,
        character_tier: Optional[CharacterTier] = None,
        is_active: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[UnityCharacterSchema]:
        """
        List Unity characters with optional filters.
        
        Args:
            actor: The requesting user
            game_id: Filter by game ID
            character_tier: Filter by character tier
            is_active: Filter by active status
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of Unity characters
        """
        logger.debug(f"📋 Listing Unity characters (game_id={game_id}, tier={character_tier}, active={is_active})")
        
        async with self.db.get_async_session() as session:
            query = select(UnityCharacter).where(
                UnityCharacter.organization_id == actor.organization_id
            )
            
            # Apply filters
            if game_id:
                query = query.where(UnityCharacter.game_id == game_id)
            if character_tier:
                query = query.where(UnityCharacter.character_tier == character_tier.value)
            if is_active is not None:
                query = query.where(UnityCharacter.is_active == is_active)
                
            # Apply pagination and ordering
            query = query.order_by(UnityCharacter.created_at.desc()).offset(offset).limit(limit)
            
            result = await session.execute(query)
            unity_characters = result.scalars().all()
            
            logger.debug(f"✅ Found {len(unity_characters)} Unity characters")
            return [self._convert_to_schema(char) for char in unity_characters]

    @trace_method
    async def update_unity_character(
        self,
        unity_character_id: str,
        character_update: UnityCharacterUpdate,
        actor: PydanticUser,
    ) -> Optional[UnityCharacterSchema]:
        """
        Update an existing Unity character.
        
        Args:
            unity_character_id: The Unity character ID
            character_update: Update data
            actor: The requesting user
            
        Returns:
            Updated Unity character if found, None otherwise
        """
        logger.info(f"📝 Updating Unity character: {unity_character_id}")
        
        async with self.db.get_async_session() as session:
            query = select(UnityCharacter).where(
                and_(
                    UnityCharacter.unity_character_id == unity_character_id,
                    UnityCharacter.organization_id == actor.organization_id,
                )
            )
            result = await session.execute(query)
            unity_character = result.scalar_one_or_none()
            
            if not unity_character:
                logger.warning(f"❌ Unity character not found for update: {unity_character_id}")
                return None

            # Update fields that were provided
            update_fields = []
            if character_update.character_name is not None:
                unity_character.character_name = character_update.character_name
                update_fields.append("character_name")
            
            if character_update.personality is not None:
                unity_character.personality = character_update.personality
                update_fields.append("personality")
                # TODO: Update the associated Letta agent's personality
            
            if character_update.backstory is not None:
                unity_character.backstory = character_update.backstory  
                update_fields.append("backstory")
                # TODO: Update the associated Letta agent's memory
            
            if character_update.voice_style is not None:
                unity_character.voice_style = character_update.voice_style
                update_fields.append("voice_style")
            
            if character_update.location is not None:
                unity_character.location = character_update.location
                update_fields.append("location")
            
            if character_update.is_active is not None:
                unity_character.is_active = character_update.is_active
                update_fields.append("is_active")
            
            if character_update.character_config is not None:
                unity_character.character_config = {
                    **(unity_character.character_config or {}),
                    **character_update.character_config
                }
                update_fields.append("character_config")

            unity_character.updated_at = datetime.utcnow()

            await session.commit()
            await session.refresh(unity_character)

            logger.info(f"✅ Updated Unity character {unity_character_id}: {', '.join(update_fields)}")
            return self._convert_to_schema(unity_character)

    @trace_method
    async def delete_unity_character(
        self,
        unity_character_id: str,
        actor: PydanticUser,
        cleanup_agent: bool = True,
    ) -> bool:
        """
        Delete a Unity character and optionally its associated Letta agent.
        
        Args:
            unity_character_id: The Unity character ID
            actor: The requesting user
            cleanup_agent: Whether to delete the associated Letta agent
            
        Returns:
            True if deleted, False if not found
        """
        logger.info(f"🗑️ Deleting Unity character: {unity_character_id} (cleanup_agent={cleanup_agent})")
        
        async with self.db.get_async_session() as session:
            query = select(UnityCharacter).where(
                and_(
                    UnityCharacter.unity_character_id == unity_character_id,
                    UnityCharacter.organization_id == actor.organization_id,
                )
            )
            result = await session.execute(query)
            unity_character = result.scalar_one_or_none()
            
            if not unity_character:
                logger.warning(f"❌ Unity character not found for deletion: {unity_character_id}")
                return False

            letta_agent_id = unity_character.letta_agent_id

            # Delete the Unity character record
            await session.delete(unity_character)
            await session.commit()

            # Cleanup the associated Letta agent if requested
            if cleanup_agent and letta_agent_id:
                try:
                    await self._cleanup_letta_agent(letta_agent_id, actor)
                    logger.info(f"✅ Cleaned up Letta agent {letta_agent_id}")
                except Exception as e:
                    logger.error(f"⚠️ Failed to cleanup Letta agent {letta_agent_id}: {e}")
                    # Don't fail the character deletion if agent cleanup fails

            logger.info(f"✅ Successfully deleted Unity character: {unity_character_id}")
            return True

    @trace_method
    async def increment_character_usage(
        self,
        unity_character_id: str,
        actor: PydanticUser,
    ) -> None:
        """
        Increment the usage statistics for a character.
        
        Args:
            unity_character_id: The Unity character ID
            actor: The requesting user
        """
        logger.debug(f"📊 Incrementing usage stats for character: {unity_character_id}")
        
        async with self.db.get_async_session() as session:
            query = select(UnityCharacter).where(
                and_(
                    UnityCharacter.unity_character_id == unity_character_id,
                    UnityCharacter.organization_id == actor.organization_id,
                )
            )
            result = await session.execute(query)
            unity_character = result.scalar_one_or_none()
            
            if unity_character:
                current_stats = unity_character.usage_stats or {}
                current_stats["interaction_count"] = current_stats.get("interaction_count", 0) + 1
                current_stats["last_interaction"] = datetime.utcnow().isoformat()
                unity_character.usage_stats = current_stats
                
                await session.commit()
                logger.debug(f"✅ Updated usage stats for {unity_character_id}")

    async def _create_letta_agent_for_character(
        self,
        character_request: UnityCharacterCreate,
        actor: PydanticUser,
    ) -> str:
        """
        Create a Letta agent for a Unity character.
        
        Args:
            character_request: Character creation request
            actor: The user creating the character
            
        Returns:
            The created Letta agent ID
        """
        # Import here to avoid circular dependency
        from letta.services.agent_manager import AgentManager
        
        agent_manager = AgentManager()
        
        # Build persona for the agent
        persona = self._build_character_persona(character_request)
        
        # Build human context (game context)
        human_context = self._build_game_context(character_request)
        
        # Determine agent type based on character tier
        agent_type = AgentType.memgpt_agent  # Default to standard MemGPT agent
        
        # Create agent request
        create_agent_request = CreateAgent(
            name=f"Unity-{character_request.character_name}-{character_request.unity_character_id}",
            description=f"Unity character: {character_request.character_name} ({character_request.role})",
            persona=persona,
            human=human_context,
            agent_type=agent_type,
            metadata={
                "unity_character_id": character_request.unity_character_id,
                "game_id": character_request.game_id,
                "character_role": character_request.role,
                "character_tier": character_request.character_tier.value,
                "created_for": "unity_integration",
            }
        )
        
        # Create the agent
        agent = await agent_manager.create_agent(create_agent_request, actor=actor)
        
        return agent.id

    async def _cleanup_letta_agent(self, letta_agent_id: str, actor: PydanticUser) -> None:
        """
        Clean up a Letta agent.
        
        Args:
            letta_agent_id: The Letta agent ID to delete
            actor: The requesting user
        """
        try:
            # Import here to avoid circular dependency
            from letta.services.agent_manager import AgentManager
            
            agent_manager = AgentManager()
            await agent_manager.delete_agent(letta_agent_id, actor=actor)
            logger.info(f"🧹 Cleaned up Letta agent: {letta_agent_id}")
        except Exception as e:
            logger.error(f"❌ Failed to cleanup Letta agent {letta_agent_id}: {e}")
            raise

    def _build_character_persona(self, character_request: UnityCharacterCreate) -> str:
        """
        Build a persona string for the Letta agent based on character data.
        
        Args:
            character_request: Character creation request
            
        Returns:
            Formatted persona string
        """
        persona_parts = [
            f"You are {character_request.character_name}, a character in the interactive story game '{character_request.game_id}'.",
            "",
            f"Your role: {character_request.role}",
            f"Your personality: {character_request.personality}",
            f"Your background: {character_request.backstory}",
        ]
        
        if character_request.voice_style:
            persona_parts.append(f"Your speaking style: {character_request.voice_style}")
            
        if character_request.location:
            persona_parts.append(f"You are typically found at: {character_request.location}")
            
        persona_parts.extend([
            "",
            "Important guidelines:",
            "- Stay in character at all times",
            "- Remember previous interactions with the player",
            "- Your responses should match your personality and background",
            "- Be consistent with the game world and story",
        ])
        
        return "\n".join(persona_parts)

    def _build_game_context(self, character_request: UnityCharacterCreate) -> str:
        """
        Build a human context string describing the game/interaction context.
        
        Args:
            character_request: Character creation request
            
        Returns:
            Formatted context string
        """
        context_parts = [
            f"You are interacting with a player in the game '{character_request.game_id}'.",
            "The player can:",
            "- Ask you questions",
            "- Have conversations with you",
            "- Request help or services",
            "- Make choices that affect the story",
            "",
            "You should:",
            "- Respond as your character would",
            "- Help advance the story when appropriate", 
            "- Remember the player's previous actions and choices",
            "- Maintain consistency with the game world",
        ]
        
        return "\n".join(context_parts)

    def _convert_to_schema(self, unity_character: UnityCharacter) -> UnityCharacterSchema:
        """
        Convert ORM model to Pydantic schema.
        
        Args:
            unity_character: The ORM model
            
        Returns:
            Pydantic schema
        """
        return UnityCharacterSchema(
            id=unity_character.id,
            unity_character_id=unity_character.unity_character_id,
            letta_agent_id=unity_character.letta_agent_id,
            character_name=unity_character.character_name,
            personality=unity_character.personality,
            backstory=unity_character.backstory,
            voice_style=unity_character.voice_style,
            role=unity_character.role,
            location=unity_character.location,
            game_id=unity_character.game_id,
            character_tier=CharacterTier(unity_character.character_tier),
            is_active=unity_character.is_active,
            character_config=unity_character.character_config,
            usage_stats=unity_character.usage_stats,
            created_at=unity_character.created_at,
            updated_at=unity_character.updated_at,
        )
