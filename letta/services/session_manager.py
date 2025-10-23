"""
Session Manager Service

Handles story session lifecycle: start, resume, restart, delete.
Manages character agents and session state.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import and_, delete, select
from sqlalchemy.exc import SQLAlchemyError

from letta.log import get_logger
from letta.orm.story import StorySession as StorySessionORM
from letta.schemas.agent import AgentType, CreateAgent
from letta.schemas.message import MessageCreate, MessageRole
from letta.schemas.story import (
    Scene,
    SessionCreate,
    SessionResume,
    SessionRestartResponse,
    SessionStartResponse,
    SessionState,
    SessionStatus,
    Story,
    StorySession,
)
from letta.schemas.user import User
from letta.server.db import db_registry
from letta.services.agent_manager import AgentManager
from letta.services.story_manager import StoryManager

logger = get_logger(__name__)


class SessionManager:
    """
    Manages story session lifecycle.
    
    Key responsibilities:
    1. Start new sessions (create agents)
    2. Resume existing sessions (restore state)
    3. Restart sessions (delete old, create new)
    4. Delete sessions (cleanup agents)
    5. Track session state
    """

    def __init__(self):
        self.db = db_registry.get_db()
        self.story_manager = StoryManager()
        self.agent_manager = AgentManager()
        logger.info("🎮 SessionManager initialized")

    async def start_session(
        self,
        session_create: SessionCreate,
        actor: User,
    ) -> SessionStartResponse:
        """
        Start a new story session.
        
        Process:
        1. Get story from database
        2. Check for existing session (delete if exists - ONE session per story)
        3. Create Letta agents for all characters
        4. Initialize session state
        5. Store in database
        
        Args:
            session_create: Session creation request
            actor: User starting the session
            
        Returns:
            Session start response with first scene
            
        Raises:
            ValueError: Story not found
            Exception: Agent creation or database errors
        """
        logger.info(
            f"🎬 Starting session for story: {session_create.story_id}, "
            f"user: {actor.id}"
        )
        
        try:
            # Step 1: Get story
            story = await self.story_manager.get_story(session_create.story_id, actor)
            if not story:
                logger.error(f"❌ Story not found: {session_create.story_id}")
                raise ValueError(f"Story '{session_create.story_id}' not found")
            
            logger.debug(f"✅ Found story: {story.title}")
            
            # Step 2: Check for existing session (ONE session per story per user)
            existing_session = await self._get_active_session(
                user_id=actor.id,
                story_id=session_create.story_id,
                actor=actor,
            )
            
            if existing_session:
                logger.warning(
                    f"⚠️ Active session exists for story {session_create.story_id}. "
                    f"Deleting old session: {existing_session.session_id}"
                )
                await self._delete_session_internal(existing_session.session_id, actor)
            
            # Step 3: Create agents for all characters
            logger.info(f"🤖 Creating agents for {len(story.characters)} characters...")
            character_agents = await self._create_character_agents(story, actor)
            logger.info(f"✅ Created {len(character_agents)} character agents")
            
            # Step 4: Initialize session state
            session_id = f"session-{uuid.uuid4()}"
            initial_state = SessionState(
                current_scene_number=1,  # Start at scene 1
                current_instruction_index=0,  # Start at first instruction
                completed_dialogue_beats=[],
                character_relationships={char.character_id: 0.0 for char in story.characters},
                player_choices=[],
                variables={},
            )
            
            # Step 5: Store in database
            async with self.db.get_async_session() as session:
                session_orm = StorySessionORM(
                    id=f"session-{uuid.uuid4()}",
                    session_id=session_id,
                    user_id=actor.id,
                    story_id=session_create.story_id,
                    status=SessionStatus.ACTIVE.value,
                    state=initial_state.dict(),
                    character_agents=character_agents,
                    metadata={
                        "story_title": story.title,
                        "total_scenes": len(story.scenes),
                        "started_at": datetime.utcnow().isoformat(),
                    },
                    organization_id=actor.organization_id,
                )
                
                session.add(session_orm)
                await session.commit()
                await session.refresh(session_orm)
                
                logger.info(f"✅ Session created: {session_id}")
                
                # Get first scene
                first_scene = story.scenes[0]
                
                # Find player character
                player_character = next(
                    (char.name for char in story.characters if char.is_main_character),
                    None,
                )
                
                return SessionStartResponse(
                    success=True,
                    session_id=session_id,
                    story_title=story.title,
                    first_scene=first_scene,
                    player_character=player_character,
                    instructions=[
                        f"Session started for '{story.title}'",
                        f"Session ID: {session_id}",
                        f"Starting scene: {first_scene.title}",
                        f"Location: {first_scene.location}",
                        f"Send dialogue with: POST /api/v1/story/sessions/{session_id}/dialogue",
                    ],
                )
        
        except ValueError as e:
            logger.error(f"❌ Session start failed (validation): {e}")
            raise
        
        except Exception as e:
            logger.error(f"❌ Session start failed (unexpected): {e}", exc_info=True)
            # Cleanup any created agents
            if 'character_agents' in locals():
                await self._cleanup_agents(character_agents, actor)
            raise Exception(f"Failed to start session: {str(e)}") from e

    async def resume_session(
        self,
        session_id: str,
        actor: User,
    ) -> SessionResume:
        """
        Resume an existing session.
        
        Args:
            session_id: Session to resume
            actor: User resuming
            
        Returns:
            Session resume data with current state
            
        Raises:
            ValueError: Session not found
        """
        logger.info(f"▶️ Resuming session: {session_id}")
        
        try:
            # Get session from database
            session_data = await self._get_session_by_id(session_id, actor)
            if not session_data:
                logger.error(f"❌ Session not found: {session_id}")
                raise ValueError(f"Session '{session_id}' not found")
            
            # Get story
            story = await self.story_manager.get_story(session_data.story_id, actor)
            if not story:
                logger.error(f"❌ Story not found: {session_data.story_id}")
                raise ValueError(f"Story '{session_data.story_id}' not found")
            
            # Get current scene
            current_scene_num = session_data.state.current_scene_number
            if current_scene_num > len(story.scenes):
                logger.warning(f"⚠️ Scene {current_scene_num} out of range, resetting to last scene")
                current_scene_num = len(story.scenes)
            
            current_scene = story.scenes[current_scene_num - 1]  # 1-indexed
            
            # TODO: Get recent interaction history from agent messages
            recent_history = []
            
            logger.info(f"✅ Session resumed: {session_id}, Scene {current_scene_num}")
            
            return SessionResume(
                session=session_data,
                current_scene=current_scene,
                recent_history=recent_history,
            )
        
        except ValueError as e:
            logger.error(f"❌ Session resume failed: {e}")
            raise
        
        except Exception as e:
            logger.error(f"❌ Session resume failed (unexpected): {e}", exc_info=True)
            raise Exception(f"Failed to resume session: {str(e)}") from e

    async def restart_session(
        self,
        session_id: str,
        actor: User,
    ) -> SessionRestartResponse:
        """
        Restart a session from the beginning.
        
        Per user requirements:
        - Wipe everything (delete agents)
        - Start fresh (new session)
        
        Args:
            session_id: Session to restart
            actor: User restarting
            
        Returns:
            Restart response with new session ID
        """
        logger.info(f"🔄 Restarting session: {session_id}")
        
        try:
            # Get existing session to know which story
            old_session = await self._get_session_by_id(session_id, actor)
            if not old_session:
                logger.error(f"❌ Session not found: {session_id}")
                raise ValueError(f"Session '{session_id}' not found")
            
            story_id = old_session.story_id
            
            # Delete old session (including agents)
            await self._delete_session_internal(session_id, actor)
            logger.info(f"✅ Old session deleted: {session_id}")
            
            # Start new session
            session_create = SessionCreate(story_id=story_id)
            start_response = await self.start_session(session_create, actor)
            
            logger.info(f"✅ Session restarted: {start_response.session_id}")
            
            return SessionRestartResponse(
                success=True,
                session_id=start_response.session_id,
                message=f"Session restarted. Old agents deleted, new session created.",
            )
        
        except ValueError as e:
            logger.error(f"❌ Session restart failed: {e}")
            raise
        
        except Exception as e:
            logger.error(f"❌ Session restart failed (unexpected): {e}", exc_info=True)
            raise Exception(f"Failed to restart session: {str(e)}") from e

    async def delete_session(
        self,
        session_id: str,
        actor: User,
    ) -> bool:
        """
        Delete a session and cleanup agents.
        
        Args:
            session_id: Session to delete
            actor: User deleting
            
        Returns:
            True if deleted, False if not found
        """
        logger.info(f"🗑️ Deleting session: {session_id}")
        return await self._delete_session_internal(session_id, actor)

    # ============================================================
    # Internal Helper Methods
    # ============================================================

    async def _create_character_agents(
        self,
        story: Story,
        actor: User,
    ) -> Dict[str, str]:
        """
        Create Letta agents for all story characters.
        
        Args:
            story: Story definition
            actor: User creating agents
            
        Returns:
            Dict of character_id -> agent_id
        """
        character_agents = {}
        
        for character in story.characters:
            try:
                logger.debug(f"  🤖 Creating agent for {character.name}...")
                
                # Build persona for the character
                persona = self._build_character_persona(character, story)
                
                # Build human context (game/story context)
                human_context = self._build_story_context(story)
                
                # Create agent
                create_agent = CreateAgent(
                    name=f"Story-{character.name}-{character.character_id}",
                    description=f"Character from story '{story.title}': {character.name}",
                    persona=persona,
                    human=human_context,
                    agent_type=AgentType.memgpt_agent,
                    metadata={
                        "story_id": story.story_id,
                        "character_id": character.character_id,
                        "character_name": character.name,
                        "is_main_character": character.is_main_character,
                        "story_role": "player" if character.is_main_character else "npc",
                    },
                )
                
                agent = await self.agent_manager.create_agent(create_agent, actor=actor)
                character_agents[character.character_id] = agent.id
                
                logger.debug(f"    ✅ Created agent {agent.id} for {character.name}")
            
            except Exception as e:
                logger.error(f"    ❌ Failed to create agent for {character.name}: {e}")
                # Cleanup already created agents
                await self._cleanup_agents(character_agents, actor)
                raise Exception(f"Failed to create agent for character '{character.name}': {str(e)}") from e
        
        return character_agents

    def _build_character_persona(self, character: StoryCharacter, story: Story) -> str:
        """Build persona string for character agent"""
        persona_parts = [
            f"You are {character.name}, a character in the story '{story.title}'.",
            "",
            f"Age: {character.age}",
            f"Gender: {character.sex}",
        ]
        
        if character.is_main_character:
            persona_parts.append("Role: You are the main character (controlled by the player)")
        else:
            persona_parts.append("Role: You are an NPC character in the story")
        
        persona_parts.extend([
            "",
            "Important guidelines:",
            "- Stay in character at all times",
            "- Remember all interactions with the player",
            "- Be consistent with the story world",
            "- Respond naturally to what the player says",
        ])
        
        return "\n".join(persona_parts)

    def _build_story_context(self, story: Story) -> str:
        """Build story context for agent"""
        context_parts = [
            f"You are in the story: {story.title}",
        ]
        
        if story.description:
            context_parts.append(f"Story description: {story.description}")
        
        context_parts.extend([
            "",
            "You are interacting with a player who is experiencing this story.",
            "Respond naturally and help bring the story to life.",
        ])
        
        return "\n".join(context_parts)

    async def _cleanup_agents(
        self,
        character_agents: Dict[str, str],
        actor: User,
    ) -> None:
        """Cleanup created agents on error"""
        logger.warning(f"🧹 Cleaning up {len(character_agents)} agents...")
        
        for char_id, agent_id in character_agents.items():
            try:
                await self.agent_manager.delete_agent(agent_id, actor=actor)
                logger.debug(f"  ✅ Deleted agent {agent_id} for {char_id}")
            except Exception as e:
                logger.error(f"  ❌ Failed to delete agent {agent_id}: {e}")

    async def _get_active_session(
        self,
        user_id: str,
        story_id: str,
        actor: User,
    ) -> Optional[StorySession]:
        """Get active session for user + story"""
        try:
            async with self.db.get_async_session() as session:
                query = select(StorySessionORM).where(
                    and_(
                        StorySessionORM.user_id == user_id,
                        StorySessionORM.story_id == story_id,
                        StorySessionORM.status == SessionStatus.ACTIVE.value,
                        StorySessionORM.organization_id == actor.organization_id,
                    )
                )
                result = await session.execute(query)
                session_orm = result.scalar_one_or_none()
                
                if session_orm:
                    return self._convert_to_schema(session_orm)
                return None
        
        except Exception as e:
            logger.error(f"❌ Error checking active session: {e}")
            return None

    async def _get_session_by_id(
        self,
        session_id: str,
        actor: User,
    ) -> Optional[StorySession]:
        """Get session by ID"""
        try:
            async with self.db.get_async_session() as session:
                query = select(StorySessionORM).where(
                    and_(
                        StorySessionORM.session_id == session_id,
                        StorySessionORM.organization_id == actor.organization_id,
                    )
                )
                result = await session.execute(query)
                session_orm = result.scalar_one_or_none()
                
                if session_orm:
                    return self._convert_to_schema(session_orm)
                return None
        
        except Exception as e:
            logger.error(f"❌ Error getting session {session_id}: {e}")
            return None

    async def _delete_session_internal(
        self,
        session_id: str,
        actor: User,
    ) -> bool:
        """Delete session and cleanup agents"""
        try:
            # Get session
            session_data = await self._get_session_by_id(session_id, actor)
            if not session_data:
                return False
            
            # Delete agents
            for char_id, agent_id in session_data.character_agents.items():
                try:
                    await self.agent_manager.delete_agent(agent_id, actor=actor)
                    logger.debug(f"  ✅ Deleted agent {agent_id} for {char_id}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Failed to delete agent {agent_id}: {e}")
            
            # Delete session from database
            async with self.db.get_async_session() as session:
                delete_stmt = delete(StorySessionORM).where(
                    and_(
                        StorySessionORM.session_id == session_id,
                        StorySessionORM.organization_id == actor.organization_id,
                    )
                )
                await session.execute(delete_stmt)
                await session.commit()
            
            logger.info(f"✅ Session deleted: {session_id}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error deleting session {session_id}: {e}", exc_info=True)
            return False

    def _convert_to_schema(self, session_orm: StorySessionORM) -> StorySession:
        """Convert ORM to Pydantic schema"""
        return StorySession(
            session_id=session_orm.session_id,
            user_id=session_orm.user_id,
            story_id=session_orm.story_id,
            status=SessionStatus(session_orm.status),
            state=SessionState(**session_orm.state),
            character_agents=session_orm.character_agents,
            created_at=session_orm.created_at,
            updated_at=session_orm.updated_at,
            completed_at=session_orm.completed_at,
        )

