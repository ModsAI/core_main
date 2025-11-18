"""
Session Manager Service

Handles story session lifecycle: start, resume, restart, delete.
Manages character agents and session state.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, delete, select
from sqlalchemy.exc import SQLAlchemyError

from letta.log import get_logger
from letta.orm.story import StorySession as StorySessionORM
from letta.schemas.agent import AgentType, CreateAgent
from letta.schemas.block import CreateBlock
from letta.schemas.message import Message, MessageCreate, MessageRole
from letta.schemas.story import (
    Scene,
    SessionCreate,
    SessionRestartResponse,
    SessionResume,
    SessionStartResponse,
    SessionState,
    SessionStatus,
    Story,
    StoryCharacter,
    StorySession,
)
from letta.schemas.user import User
from letta.server.db import db_registry
from letta.services.agent_manager import AgentManager
from letta.services.block_manager import BlockManager
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
        self.story_manager = StoryManager()
        self.agent_manager = AgentManager()
        self.block_manager = BlockManager()
        logger.info("🎮 SessionManager initialized")

    # ============================================================
    # Relationship Array Helper Methods
    # ============================================================

    @staticmethod
    def _find_relationship_point(points_array: List, rel_id: str) -> Optional[int]:
        """Find points value for a relationship ID in array"""
        for item in points_array:
            if isinstance(item, dict) and item.get("id") == rel_id:
                return item.get("points")
            elif hasattr(item, "id") and item.id == rel_id:
                return item.points
        return None

    @staticmethod
    def _find_relationship_level(levels_array: List, rel_id: str) -> Optional[int]:
        """Find level value for a relationship ID in array"""
        for item in levels_array:
            if isinstance(item, dict) and item.get("id") == rel_id:
                return item.get("level")
            elif hasattr(item, "id") and item.id == rel_id:
                return item.level
        return None

    @staticmethod
    def _update_relationship_point(points_array: List, rel_id: str, new_points: int) -> None:
        """Update or add relationship points in array"""
        for item in points_array:
            if (isinstance(item, dict) and item.get("id") == rel_id) or \
               (hasattr(item, "id") and item.id == rel_id):
                if isinstance(item, dict):
                    item["points"] = new_points
                else:
                    item.points = new_points
                return
        # Not found - add new entry
        from letta.schemas.story import RelationshipPoint
        points_array.append(RelationshipPoint(id=rel_id, points=new_points))

    @staticmethod
    def _update_relationship_level(levels_array: List, rel_id: str, new_level: int) -> None:
        """Update or add relationship level in array"""
        for item in levels_array:
            if (isinstance(item, dict) and item.get("id") == rel_id) or \
               (hasattr(item, "id") and item.id == rel_id):
                if isinstance(item, dict):
                    item["level"] = new_level
                else:
                    item.level = new_level
                return
        # Not found - add new entry
        from letta.schemas.story import RelationshipLevel
        levels_array.append(RelationshipLevel(id=rel_id, level=new_level))

    # ============================================================
    # Session Lifecycle Methods
    # ============================================================

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
        logger.info(f"🎬 Starting session for story: {session_create.story_id}, " f"user: {actor.id}")

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
                    f"⚠️ Active session exists for story {session_create.story_id}. " f"Deleting old session: {existing_session.session_id}"
                )
                await self._delete_session_internal(existing_session.session_id, actor)

            # Step 3: Initialize session state FIRST (needed for agent memory)
            session_id = f"session-{uuid.uuid4()}"

            # Build character relationships (only for characters with character_id)
            character_relationships = {char.character_id: 0.0 for char in story.characters if char.character_id is not None}

            # Initialize relationship system (NEW - for multi-track relationships)
            from letta.schemas.story import RelationshipPoint, RelationshipLevel
            
            relationship_points = []
            relationship_levels = []
            
            if story.relationships:
                for rel in story.relationships:
                    rel_id = rel.relationship_id
                    starting_points = rel.starting_points
                    # Calculate starting level from starting points (1-based)
                    level = (starting_points // rel.points_per_level) + 1 if rel.points_per_level > 0 else 1
                    level = min(level, rel.max_levels)
                    level = max(level, 1)  # Floor at 1, not 0
                    
                    relationship_points.append(RelationshipPoint(id=rel_id, points=starting_points))
                    relationship_levels.append(RelationshipLevel(id=rel_id, level=level))
                    logger.debug(f"  💝 Initialized relationship {rel_id}: {starting_points} points, level {level}")

            initial_state = SessionState(
                current_scene_number=1,  # Start at scene 1
                current_instruction_index=0,  # Start at first instruction
                completed_dialogue_beats=[],
                character_relationships=character_relationships,  # Legacy - kept for backwards compatibility
                relationship_points=relationship_points,  # NEW - multi-track relationships
                relationship_levels=relationship_levels,  # NEW - multi-track relationships
                player_choices=[],
                variables={},
            )
            
            logger.debug(f"✅ Session state initialized with {len(relationship_points)} relationships")

            # Step 4: Create agents with relationship context
            logger.info(f"🤖 Creating agents for {len(story.characters)} characters...")
            character_agents = await self._create_character_agents(story, actor, initial_state)
            logger.info(f"✅ Created {len(character_agents)} character agents with relationship awareness")

            # Step 5: Store in database
            async with db_registry.async_session() as session:
                async with session.begin():
                    session_orm = StorySessionORM(
                        id=f"session-{uuid.uuid4()}",
                        session_id=session_id,
                        user_id=actor.id,
                        story_id=session_create.story_id,
                        status=SessionStatus.ACTIVE.value,
                        state=initial_state.dict(),
                        character_agents=character_agents,
                        session_metadata={
                            "story_title": story.title,
                            "total_scenes": len(story.scenes),
                            "started_at": datetime.utcnow().isoformat(),
                        },
                        organization_id=actor.organization_id,
                    )

                    session.add(session_orm)
                    await session.flush()
                    await session.refresh(session_orm)

                logger.info(f"✅ Session created: {session_id}")

                # Get first scene
                first_scene = story.scenes[0]

                # Find player character (the one they're playing as)
                player_character = next(
                    (char.name for char in story.characters if char.is_main_character),
                    None,
                )

                # 🎮 FIRST-PERSON MODE: Get NPCs only (exclude main character)
                # Player can only talk TO other characters, not to themselves
                available_characters = [char.character_id for char in story.characters if char.character_id and not char.is_main_character]

                logger.info(f"✅ Available characters: {', '.join(available_characters)}")

                return SessionStartResponse(
                    success=True,
                    session_id=session_id,
                    story_title=story.title,
                    first_scene=first_scene,
                    current_scene=first_scene,  # Alias for compatibility
                    player_character=player_character,
                    available_characters=available_characters,
                    instructions=[
                        f"Session started for '{story.title}'",
                        f"Session ID: {session_id}",
                        f"Starting scene: {first_scene.title}",
                        f"Location: {first_scene.location}",
                        f"Characters: {', '.join(available_characters)}",
                        f"Send dialogue with: POST /api/v1/story/sessions/{session_id}/dialogue",
                    ],
                )

        except ValueError as e:
            logger.error(f"❌ Session start failed (validation): {e}")
            raise

        except Exception as e:
            logger.error(f"❌ Session start failed (unexpected): {e}", exc_info=True)
            # Cleanup any created agents
            if "character_agents" in locals():
                await self._cleanup_agents(character_agents, actor)
            raise Exception(f"Failed to start session: {str(e)}") from e

    async def resume_session(
        self,
        story_id: str,  # Changed: now accepts story_id to find active session
        actor: User,
        server: "SyncServer",
    ) -> SessionResume:
        """
        Resume an existing session for a story.

        Finds and resumes the active session for the given story.

        Args:
            story_id: Story ID to find active session for
            actor: User resuming
            server: Server instance for accessing message manager

        Returns:
            Session resume data with current state

        Raises:
            ValueError: No active session found for this story
        """
        logger.info(f"▶️ Resuming session for story: {story_id}")

        try:
            # Find active session for this story
            session_data = await self._get_active_session(actor.id, story_id, actor)
            if not session_data:
                logger.error(f"❌ No active session found for story: {story_id}")
                raise ValueError(f"No active session found for story '{story_id}'. Start a new session first.")

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

            # Get recent interaction history using generic method
            recent_history = await self.get_recent_messages(
                session_id=session_data.session_id,
                limit=10,  # Last 10 messages
                actor=actor,
                server=server,
            )

            logger.info(f"✅ Session resumed: {session_data.session_id}, Scene {current_scene_num}")

            return SessionResume(
                success=True,
                session_id=session_data.session_id,
                story_title=story.title,
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

    async def get_recent_messages(
        self,
        session_id: str,
        limit: int,
        actor: User,
        server: "SyncServer",
    ) -> List[Dict[str, Any]]:
        """
        Get recent dialogue messages from a session.
        
        Retrieves last N messages from all character agents,
        formats them as simple dialogue, and returns in chronological order.
        
        Args:
            session_id: Session identifier
            limit: Max number of messages to return
            actor: User making request
            server: Server instance for accessing message manager
            
        Returns:
            List of message dictionaries with character, message, timestamp, role
        """
        logger.info(f"📜 Getting recent messages for session: {session_id} (limit={limit})")
        
        # Get session
        session_data = await self._get_session_by_id(session_id, actor)
        if not session_data:
            raise ValueError(f"Session '{session_id}' not found")
        
        # Collect messages from all character agents
        all_messages = []
        
        for char_id, agent_id in session_data.character_agents.items():
            try:
                # Get messages for this agent
                messages = await server.message_manager.list_messages_for_agent_async(
                    agent_id=agent_id,
                    actor=actor,
                    limit=limit * 2,  # Get extra to ensure we have enough after filtering
                    ascending=False,  # Newest first
                )
                
                logger.info(f"📨 Agent {char_id}: Retrieved {len(messages)} messages")
                
                # Extract dialogue from messages
                for msg in messages:
                    logger.info(f"  Message role: {msg.role}, id: {msg.id}")
                    
                    # Handle assistant messages (NPC dialogue)
                    if msg.role == MessageRole.assistant:
                        dialogue_text = self._extract_dialogue_text(msg)
                        if dialogue_text:
                            logger.info(f"    ✅ Extracted dialogue: {dialogue_text[:50]}...")
                            all_messages.append({
                                "character": char_id,
                                "message": dialogue_text,
                                "timestamp": msg.created_at.isoformat() if msg.created_at else "",
                                "role": "assistant"
                            })
                        else:
                            logger.info(f"    ❌ No dialogue extracted from assistant message")
                    
                    # Handle user messages (player input)
                    elif msg.role == MessageRole.user:
                        user_text = self._extract_user_text(msg)
                        if user_text:
                            logger.info(f"    ✅ Extracted user text: {user_text[:50]}...")
                            all_messages.append({
                                "character": "player",
                                "message": user_text,
                                "timestamp": msg.created_at.isoformat() if msg.created_at else "",
                                "role": "user"
                            })
                        else:
                            logger.info(f"    ❌ No text extracted from user message")
            
            except Exception as e:
                logger.warning(f"⚠️ Failed to get messages for agent {agent_id} (char: {char_id}): {e}")
                continue
        
        # Sort by timestamp (chronological order)
        all_messages.sort(key=lambda x: x["timestamp"])
        
        # Return last N messages
        recent = all_messages[-limit:] if all_messages else []
        logger.info(f"✅ Retrieved {len(recent)} recent messages from {len(session_data.character_agents)} agents")
        return recent

    def _extract_dialogue_text(self, message: Message) -> Optional[str]:
        """Extract dialogue text from assistant message (NPC speaking)."""
        import json
        
        # Check for send_message tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                if hasattr(tool_call, 'function') and tool_call.function.name == 'send_message':
                    try:
                        args = json.loads(tool_call.function.arguments)
                        if 'message' in args:
                            return args['message']
                    except (json.JSONDecodeError, AttributeError):
                        continue
        
        return None

    def _extract_user_text(self, message: Message) -> Optional[str]:
        """Extract text from user message (player speaking)."""
        # User messages typically have content as list of TextContent
        if hasattr(message, 'content') and message.content:
            if isinstance(message.content, list):
                for item in message.content:
                    if hasattr(item, 'text'):
                        return item.text
            elif isinstance(message.content, str):
                return message.content
        
        return None

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
        session_state: SessionState,  # NEW: Pass session state for relationship context
    ) -> Dict[str, str]:
        """
        Create Letta agents for story characters.

        🎮 FIRST-PERSON MODE:
        - Main character (isMainCharacter: true) is controlled by the player
        - We only create agents for NPCs (non-main characters)
        - Player IS the main character, not talking TO them

        Args:
            story: Story definition
            actor: User creating agents
            session_state: Current session state for relationship context

        Returns:
            Dict of character name -> agent_id (excludes main character)
        """
        character_agents = {}

        for character in story.characters:
            # 🎮 FIRST-PERSON MODE: Skip main character (player controls them)
            if character.is_main_character:
                logger.info(f"  ⏭️ Skipping main character (player-controlled): {character.name}")
                continue

            try:
                logger.debug(f"  🤖 Creating agent for NPC: {character.name}...")

                # ============================================================
                # Build Core Memory Blocks (Letta's persistent memory system)
                # ============================================================

                # Block 1: persona - Character identity and guidelines
                persona_value = self._build_character_persona(character, story)

                # Block 2: human - Player info and story context
                human_value = self._build_story_context(story)

                # Block 3: current_scene - Current scene context (starts with Scene 1)
                # Get first scene to initialize scene context
                first_scene = story.scenes[0] if story.scenes else None
                if first_scene:
                    scene_value = self._build_scene_context(first_scene)
                else:
                    # Fallback if no scenes exist
                    scene_value = "=== CURRENT SCENE ===\nThe story is about to begin..."

                # Block 4: relationships - Current relationship status (NEW!)
                relationships_value = self._build_relationships_context(story, session_state, character)

                # Create memory blocks (NOW WITH 4TH BLOCK!)
                memory_blocks = [
                    CreateBlock(
                        label="persona",
                        value=persona_value,
                        limit=2000,  # Character limit for block
                    ),
                    CreateBlock(
                        label="human",
                        value=human_value,
                        limit=2000,
                    ),
                    CreateBlock(
                        label="current_scene",
                        value=scene_value,
                        limit=1000,
                    ),
                    CreateBlock(
                        label="relationships",  # NEW: 4th memory block for relationship awareness!
                        value=relationships_value,
                        limit=1500,  # Enough space for multiple relationships
                    ),
                ]

                logger.debug(f"    📝 Created {len(memory_blocks)} core memory blocks for {character.name} (including relationships)")

                # ============================================================
                # Create Agent with Core Memory Blocks
                # ============================================================

                # Use Gemini 2.0 Flash (latest model) for fast, quality dialogue generation
                # Can be overridden via environment variable DEFAULT_STORY_MODEL
                import os

                from letta.schemas.embedding_config import EmbeddingConfig
                from letta.schemas.llm_config import LLMConfig

                default_model = os.getenv("DEFAULT_STORY_MODEL", "gemini-2.0-flash-001")

                # Build LLM config for Gemini 2.0 Flash
                llm_config = LLMConfig(
                    model="gemini-2.0-flash-001",  # Google API expects model name without provider prefix
                    model_endpoint_type="google_ai",
                    context_window=1000000,  # 1M token context window!
                )

                # Build embedding config (use Letta's free embedding service)
                embedding_config = EmbeddingConfig(
                    embedding_endpoint_type="hugging-face",
                    embedding_endpoint="https://embeddings.memgpt.ai",
                    embedding_model="letta-free",
                    embedding_dim=1024,
                    embedding_chunk_size=300,
                )

                create_agent = CreateAgent(
                    name=f"Story-{character.name}-{character.character_id}",
                    description=f"Character from story '{story.title}': {character.name}",
                    memory_blocks=memory_blocks,  # ✅ NEW: Core memory blocks!
                    agent_type=AgentType.memgpt_agent,
                    llm_config=llm_config,
                    embedding_config=embedding_config,
                    metadata={
                        "story_id": story.story_id,
                        "character_id": character.character_id,
                        "character_name": character.name,
                        "is_main_character": character.is_main_character,
                        "story_role": "player" if character.is_main_character else "npc",
                    },
                )

                agent = await self.agent_manager.create_agent_async(create_agent, actor=actor)
                # Use character.name (not character_id) for user-friendly character identification
                character_agents[character.name] = agent.id

                logger.debug(f"    ✅ Created agent {agent.id} for {character.name}")

            except Exception as e:
                logger.error(f"    ❌ Failed to create agent for {character.name}: {e}")
                # Cleanup already created agents
                await self._cleanup_agents(character_agents, actor)
                raise Exception(f"Failed to create agent for character '{character.name}': {str(e)}") from e

        return character_agents

    def _build_story_narrative(self, story: Story) -> str:
        """
        Build concise story narrative for character awareness (under 600 chars).
        """
        # Concise character list (names only)
        cast_list = ", ".join([c.name for c in story.characters[:8]])  # Max 8 chars
        if len(story.characters) > 8:
            cast_list += f" +{len(story.characters) - 8} more"

        # Concise scene list (titles only)
        scene_list = " → ".join([s.title for s in story.scenes[:5]])  # Max 5 scenes
        if len(story.scenes) > 5:
            scene_list += f" +{len(story.scenes) - 5} more"

        return f"STORY: {story.title}\nCAST: {cast_list}\nSCENES: {scene_list}"

    def _build_relationships_context(
        self, 
        story: Story, 
        session_state: SessionState,
        current_character: StoryCharacter
    ) -> str:
        """
        Build relationship status context for AI agent memory.
        
        This block allows the AI to be aware of relationship levels and adjust
        dialogue/behavior accordingly. Updated in real-time as relationships change.
        
        Args:
            story: Story with relationship definitions
            session_state: Current session state with relationship values
            current_character: The character this agent represents
            
        Returns:
            Formatted relationship context string
        """
        if not story.relationships:
            return "=== RELATIONSHIPS ===\nNo relationship tracking for this story."
        
        # Find main character name for context
        main_character = next((char for char in story.characters if char.is_main_character), None)
        player_name = main_character.name if main_character else "the player"
        
        context_parts = [
            "=== RELATIONSHIP STATUS ===",
            f"Your relationship with {player_name}:",
            ""
        ]
        
        # Find relationships involving this character
        char_relationships = [
            rel for rel in story.relationships 
            if rel.character.lower() == current_character.name.lower()
        ]
        
        if not char_relationships:
            context_parts.append(f"No tracked relationships for {current_character.name}")
            return "\n".join(context_parts)
        
        # Add each relationship with current status
        for rel in char_relationships:
            rel_id = rel.relationship_id
            current_points = self._find_relationship_point(session_state.relationship_points, rel_id)
            if current_points is None:
                current_points = rel.starting_points
            current_level = self._find_relationship_level(session_state.relationship_levels, rel_id)
            if current_level is None:
                current_level = 1  # Default to level 1 (1-based)
            
            # Calculate progress to next level
            if rel.points_per_level > 0:
                points_in_level = current_points % rel.points_per_level
                progress_pct = int((points_in_level / rel.points_per_level) * 100)
            else:
                progress_pct = 0
            
            # Format relationship type nicely
            rel_type_display = rel.type.capitalize()
            
            context_parts.extend([
                f"• {rel_type_display}:",
                f"  - Level: {current_level}/{rel.max_levels}",
                f"  - Points: {current_points} ({progress_pct}% to next level)",
                f"  - Status: {self._get_relationship_description(rel.type, current_level, rel.max_levels)}",
                ""
            ])
        
        context_parts.extend([
            "IMPORTANT:",
            "- Adjust your tone and familiarity based on relationship level",
            "- Higher friendship = more casual, warm, trusting",
            "- Higher romance = more flirty, caring, intimate",
            "- Lower levels = more formal, cautious, distant",
            "- Let relationship level naturally influence your responses"
        ])
        
        return "\n".join(context_parts)
    
    def _get_relationship_description(self, rel_type: str, level: int, max_levels: int) -> str:
        """Get human-readable relationship status description."""
        progress = level / max_levels if max_levels > 0 else 0
        
        descriptions = {
            "friendship": [
                "Strangers",
                "Acquaintances", 
                "Friendly",
                "Close Friends",
                "Best Friends"
            ],
            "romance": [
                "No Interest",
                "Curious",
                "Attracted",
                "Dating",
                "In Love"
            ],
            "rivalry": [
                "Neutral",
                "Competitive",
                "Hostile",
                "Enemies",
                "Arch-Nemesis"
            ],
            "mentorship": [
                "No Relationship",
                "Student",
                "Apprentice",
                "Protégé",
                "Master & Student"
            ]
        }
        
        labels = descriptions.get(rel_type, ["Unknown"] * 5)
        index = min(int(progress * len(labels)), len(labels) - 1)
        return labels[index]

    def _build_character_persona(self, character: StoryCharacter, story: Story) -> str:
        """
        Build rich persona string for character agent.

        Includes:
        - Character identity (name, age, gender, role)
        - Relationship to main character
        - Story world context
        - FULL STORY NARRATIVE (all scenes, characters, plot)
        - Behavioral guidelines
        """
        # Find main character for relationship context
        main_character = next((char for char in story.characters if char.is_main_character), None)

        persona_parts = [
            f"You are {character.name}, a character in the story '{story.title}'.",
            "",
            "IDENTITY:",
            f"• Name: {character.name}",
            f"• Age: {character.age}",
            f"• Gender: {character.sex}",
        ]

        # Add role
        if character.is_main_character:
            persona_parts.append("• Role: Main character (controlled by the player)")
        else:
            persona_parts.append("• Role: Supporting character (NPC)")

        # Add relationship to main character (if not the main character)
        if not character.is_main_character and main_character:
            persona_parts.extend(
                [
                    "",
                    "RELATIONSHIP:",
                    f"• You are interacting with {main_character.name}, the main character",
                    f"• {main_character.name} is the player (they are experiencing this story through {main_character.name})",
                ]
            )

        # Add story context
        if story.description:
            persona_parts.extend(
                [
                    "",
                    "STORY WORLD:",
                    f"• Setting: {story.description}",
                    "• You are part of this interactive story experience",
                ]
            )

        # ⭐ NEW: Add full story narrative (all scenes, characters, plot)
        persona_parts.extend(
            [
                "",
                self._build_story_narrative(story),
            ]
        )

        # Add behavioral guidelines (from Letta's best practices)
        persona_parts.extend(
            [
                "",
                "IMPORTANT GUIDELINES:",
                "• Stay in character at all times - you ARE this character",
                "• Remember all interactions and conversations",
                "• Be consistent with the story world and your role",
                "• Respond naturally and authentically to what the player says",
                "• React emotionally and meaningfully based on your character",
            ]
        )

        return "\n".join(persona_parts)

    def _build_story_context(self, story: Story) -> str:
        """
        Build story context for agent (human block).

        Tells NPCs:
        - Who the player is (main character name)
        - Story title and description
        - How to interact with the player

        This is the 'human' block in Letta's core memory system.
        """
        # Find main character (who the player is)
        main_character = next((char for char in story.characters if char.is_main_character), None)

        context_parts = []

        # Section 1: Player Information
        if main_character:
            context_parts.extend(
                [
                    "=== PLAYER INFORMATION ===",
                    f"You are interacting with {main_character.name}, the main character of this story.",
                    f"{main_character.name} is the player - they are experiencing the story through {main_character.name}'s perspective.",
                    "",
                ]
            )
        else:
            context_parts.extend(
                [
                    "=== PLAYER INFORMATION ===",
                    "You are interacting with the player who is experiencing this story.",
                    "",
                ]
            )

        # Section 2: Story Context
        context_parts.extend(
            [
                "=== STORY CONTEXT ===",
                f"Story Title: {story.title}",
            ]
        )

        if story.description:
            context_parts.append(f"Description: {story.description}")

        # Section 3: Interaction Guidelines
        context_parts.extend(
            [
                "",
                "=== INTERACTION GUIDELINES ===",
                f"• Respond naturally to what {main_character.name if main_character else 'the player'} says",
                "• Stay in character and bring the story to life",
                "• Remember your relationship and history with this character",
                "• Be authentic and emotionally responsive",
            ]
        )

        return "\n".join(context_parts)

    def _build_scene_context(self, scene: Scene) -> str:
        """
        Build current scene context for agent (current_scene block).

        This block gets updated when scenes change during gameplay.

        Provides:
        - Scene number and title
        - Location/setting
        - Mood/atmosphere
        - Brief context (minimal, useful)

        Args:
            scene: Current scene object

        Returns:
            Formatted scene context string
        """
        context_parts = [
            "=== CURRENT SCENE ===",
            f"Scene {scene.scene_number}: {scene.title}",
            "",
        ]

        # Add location
        if scene.location:
            context_parts.append(f"Location: {scene.location}")

        # Extract mood from scene (if available in instructions)
        # Look for mood indicators in the setting description
        mood_indicators = []
        if "mood:" in scene.location.lower():
            # Extract mood from "Mood: desperate, fearful" format
            mood_part = scene.location.lower().split("mood:")[-1].strip()
            mood_indicators.append(f"Mood: {mood_part}")

        if mood_indicators:
            context_parts.extend(mood_indicators)

        # Add minimal context
        context_parts.extend(
            [
                "",
                "You are currently in this scene of the story.",
                "Respond naturally to what happens and what the player says.",
            ]
        )

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
            async with db_registry.async_session() as session:
                async with session.begin():
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
            async with db_registry.async_session() as session:
                async with session.begin():
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
        # Q7 DEBUG: Log what we're deserializing
        logger.debug(f"🔍 Converting ORM to schema - state keys: {session_orm.state.keys()}")
        logger.debug(f"🔍 completed_narration_beats in state: {session_orm.state.get('completed_narration_beats', 'KEY NOT FOUND')}")

        # Deserialize state
        session_state = SessionState(**session_orm.state)
        logger.debug(f"🔍 After SessionState init: {len(session_state.completed_narration_beats)} narration beats")

        return StorySession(
            session_id=session_orm.session_id,
            user_id=session_orm.user_id,
            story_id=session_orm.story_id,
            status=SessionStatus(session_orm.status),
            state=session_state,
            character_agents=session_orm.character_agents,
            created_at=session_orm.created_at,
            updated_at=session_orm.updated_at,
            completed_at=session_orm.completed_at,
            version=session_orm.version,  # Include version for optimistic locking
        )

    # ============================================================
    # NEW: Kon's Unity Integration - GET Endpoint Helpers
    # ============================================================

    def _get_next_instruction(
        self,
        story: Story,
        session_state: SessionState,
    ) -> Optional[Dict]:
        """
        Get the next instruction/beat from the story.

        Returns the next dialogue beat or instruction that needs to be addressed.
        This is what Unity needs to display to the player.

        NOW WITH Q4 SUPPORT:
        - Checks beat dependencies (requires_beats)
        - Respects beat priority (required vs optional)
        - Skips beats with unsatisfied dependencies

        Args:
            story: The story object
            session_state: Current session state

        Returns:
            Dictionary with next instruction info, or None if story complete
        """
        current_scene_num = session_state.current_scene_number

        # Check if story is complete
        if current_scene_num > len(story.scenes):
            return {
                "type": "end",
                "beat_id": None,
                "beat_number": None,
                "global_beat_number": None,
                "character": None,
                "model": None,
                "topic": "Story Complete",
                "script_text": "The story has ended.",
                "is_completed": True,
                "priority": None,
                "requires_beats": [],
                "instruction_details": None,
            }

        # Get current scene
        current_scene = story.scenes[current_scene_num - 1]

        # Q5: Collect all beat types with their completion status
        all_beats = []

        # Add dialogue beats
        for beat in current_scene.dialogue_beats:
            all_beats.append({**beat, "beat_type": "dialogue", "completed": beat.get("beat_id") in session_state.completed_dialogue_beats})

        # Q5: Add narration beats
        for beat in current_scene.narration_beats:
            all_beats.append(
                {**beat, "beat_type": "narration", "completed": beat.get("beat_id") in session_state.completed_narration_beats}
            )

        # Q5: Add action beats
        for beat in current_scene.action_beats:
            all_beats.append({**beat, "beat_type": "action", "completed": beat.get("beat_id") in session_state.completed_action_beats})

        # Sort by global_beat_number to maintain story order
        all_beats.sort(key=lambda b: b.get("global_beat_number", 999))

        logger.debug(
            f"  📋 Scene {current_scene_num} has {len(all_beats)} total beats (D:{len(current_scene.dialogue_beats)}, N:{len(current_scene.narration_beats)}, A:{len(current_scene.action_beats)})"
        )

        # Q4 & Q5: Find next available beat (considering dependencies and completion)
        for idx, beat in enumerate(all_beats):
            beat_id = beat.get("beat_id")
            beat_type = beat.get("beat_type")

            if beat_id and not beat.get("completed"):
                # Q4: Check if dependencies are satisfied
                requires_beats = beat.get("requires_beats", [])
                all_completed = (
                    session_state.completed_dialogue_beats + session_state.completed_narration_beats + session_state.completed_action_beats
                )
                dependencies_met = all(req_beat in all_completed for req_beat in requires_beats)

                if not dependencies_met:
                    # Dependencies not met - skip this beat for now
                    priority = beat.get("priority", "required")
                    logger.debug(f"  ⏭️ Skipping {beat_type} {beat_id} - dependencies not met: {requires_beats} " f"(priority: {priority})")
                    continue  # Try next beat

                # Dependencies satisfied! Return this beat
                global_beat_number = beat.get("global_beat_number")
                priority = beat.get("priority", "required")

                # Build response based on beat type
                if beat_type == "dialogue":
                    character_name = beat.get("character", "Unknown")
                    topic = beat.get("topic", "conversation")
                    script_text = beat.get("script_text", "")

                    # Look up character to get model
                    character_obj = next(
                        (char for char in story.characters if char.name == character_name),
                        None
                    )

                    # Extract keywords from topic and script
                    keywords = []
                    if topic:
                        keywords.extend(topic.lower().split()[:3])
                    if script_text:
                        words = [w.strip(".,!?") for w in script_text.lower().split() if len(w) > 4]
                        keywords.extend(words[:3])

                    return {
                        "type": "dialogue",
                        "beat_id": beat_id,
                        "beat_number": beat.get("beat_number"),
                        "global_beat_number": global_beat_number,
                        "character": character_name,
                        "model": character_obj.model if character_obj else None,
                        "topic": topic,
                        "script_text": script_text,
                        "is_completed": False,
                        "priority": priority,
                        "requires_beats": requires_beats,
                        "instruction_details": {
                            "general_guidance": f"Guide conversation toward: {topic}",
                            "emotional_tone": beat.get("emotion"),
                            "keywords": list(set(keywords)),
                        },
                        # Q9: Metadata enrichment
                        "emotion": beat.get("emotion"),
                        "animation": beat.get("animation"),
                        "camera_angle": beat.get("camera_angle"),
                        "timing_hint": beat.get("timing_hint"),
                        "sfx": beat.get("sfx"),
                        "music_cue": beat.get("music_cue"),
                        # Multiple choice support
                        "choices": beat.get("choices"),
                    }

                elif beat_type == "narration":
                    text = beat.get("text", "")
                    logger.info(
                        f"DEBUG _get_next_instruction: beat_type=narration, beat_id={beat_id}, beat keys={list(beat.keys())}, beat['choices']={beat.get('choices')}"
                    )
                    return {
                        "type": "narration",
                        "beat_id": beat_id,
                        "beat_number": beat.get("beat_number"),
                        "global_beat_number": global_beat_number,
                        "character": None,
                        "model": None,
                        "topic": "Narration",
                        "script_text": text,
                        "is_completed": False,
                        "priority": priority,
                        "requires_beats": requires_beats,
                        "instruction_details": {
                            "general_guidance": "Display narration to player",
                            "emotional_tone": None,
                            "keywords": [],
                        },
                        # Q9: Metadata enrichment
                        "emotion": None,
                        "animation": None,
                        "camera_angle": beat.get("camera_angle"),
                        "timing_hint": beat.get("timing_hint"),
                        "sfx": beat.get("sfx"),
                        "music_cue": beat.get("music_cue"),
                        # Multiple choice support
                        "choices": beat.get("choices"),
                    }

                elif beat_type == "action":
                    character_name = beat.get("character", "Unknown")
                    action_text = beat.get("action_text", "")
                    
                    # Look up character to get model
                    character_obj = next(
                        (char for char in story.characters if char.name == character_name),
                        None
                    )
                    
                    return {
                        "type": "action",
                        "beat_id": beat_id,
                        "beat_number": beat.get("beat_number"),
                        "global_beat_number": global_beat_number,
                        "character": character_name,
                        "model": character_obj.model if character_obj else None,
                        "topic": "Action",
                        "script_text": action_text,
                        "is_completed": False,
                        "priority": priority,
                        "requires_beats": requires_beats,
                        "instruction_details": {
                            "general_guidance": f"Display action: {action_text}",
                            "emotional_tone": None,
                            "keywords": [],
                        },
                        # Q9: Metadata enrichment
                        "emotion": None,
                        "animation": beat.get("animation"),
                        "camera_angle": beat.get("camera_angle"),
                        "timing_hint": beat.get("timing_hint"),
                        "sfx": beat.get("sfx"),
                        "music_cue": None,
                        # Multiple choice support
                        "choices": beat.get("choices"),
                    }

        # All available beats completed in this scene
        # (Either truly complete, or remaining beats have unsatisfied dependencies)
        return {
            "type": "setting",
            "beat_id": None,
            "beat_number": None,
            "global_beat_number": None,
            "character": None,
            "model": None,
            "topic": "Scene Transition",
            "script_text": f"All available dialogue beats completed in {current_scene.title}. Ready to advance to next scene.",
            "is_completed": True,
            "priority": None,
            "requires_beats": [],
            "instruction_details": {
                "general_guidance": "Scene complete, ready for next scene",
                "emotional_tone": "transitional",
                "keywords": ["complete", "next", "advance"],
            },
        }

    def _get_current_beat_info(
        self,
        story: Story,
        session_state: SessionState,
    ) -> Dict:
        """
        Get information about the current beat progress.

        Returns details about completed vs remaining beats in current scene.

        Args:
            story: The story object
            session_state: Current session state

        Returns:
            Dictionary with beat progress info
        """
        current_scene_num = session_state.current_scene_number

        # Handle story completion
        if current_scene_num > len(story.scenes):
            return {
                "beats_completed": session_state.completed_dialogue_beats,
                "beats_remaining": [],
                "total_beats_in_scene": 0,
                "scene_progress": 1.0,
                "scene_complete": True,
            }

        current_scene = story.scenes[current_scene_num - 1]
        completed_beats = session_state.completed_dialogue_beats

        # Get all beat IDs in current scene
        all_beat_ids = [beat.get("beat_id") for beat in current_scene.dialogue_beats if beat.get("beat_id")]

        # Separate completed vs remaining
        completed_in_scene = [bid for bid in all_beat_ids if bid in completed_beats]
        remaining_in_scene = [bid for bid in all_beat_ids if bid not in completed_beats]

        total_beats = len(all_beat_ids)
        completed_count = len(completed_in_scene)

        # Calculate progress
        scene_progress = completed_count / total_beats if total_beats > 0 else 0.0
        scene_complete = len(remaining_in_scene) == 0

        return {
            "beats_completed": completed_in_scene,
            "beats_remaining": remaining_in_scene,
            "total_beats_in_scene": total_beats,
            "scene_progress": scene_progress,
            "scene_complete": scene_complete,
        }

    async def get_session_state(
        self,
        session_id: str,
        actor: User,
    ):
        """
        Get comprehensive session state for Unity integration.

        This is the main method that Kon's server will call to get:
        - Current scene/setting
        - Available characters
        - Next instruction/beat
        - Progress tracking

        Args:
            session_id: Session identifier
            actor: User making the request

        Returns:
            SessionStateResponse with all state info
        """
        from letta.schemas.story import CharacterInfo, CurrentSettingInfo, NextInstructionInfo, ProgressInfo, SessionStateResponse

        # Get session
        session = await self._get_session_by_id(session_id, actor)
        if not session:
            raise ValueError(f"Session '{session_id}' not found")

        # Get story
        story = await self.story_manager.get_story(session.story_id, actor)
        if not story:
            raise ValueError(f"Story '{session.story_id}' not found")

        # Get current scene
        current_scene_num = session.state.current_scene_number
        if current_scene_num > len(story.scenes):
            # Story complete
            current_scene = story.scenes[-1]  # Last scene
        else:
            current_scene = story.scenes[current_scene_num - 1]

        # Get current instruction to extract location
        current_instruction_index = session.state.current_instruction_index
        current_instruction = None
        location = "Unknown location"  # Default fallback
        
        if current_instruction_index < len(current_scene.instructions):
            current_instruction = current_scene.instructions[current_instruction_index]
            # Extract location from instruction
            # The field is called 'setting' in StoryInstruction schema, not 'location'
            if isinstance(current_instruction, dict):
                location = current_instruction.get("setting") or current_instruction.get("location", "Unknown location")
            elif hasattr(current_instruction, "setting"):
                location = current_instruction.setting or "Unknown location"
            elif hasattr(current_instruction, "location"):
                location = current_instruction.location or "Unknown location"

        # Build current setting info
        current_setting = CurrentSettingInfo(
            scene_id=current_scene.scene_id,
            scene_number=current_scene.scene_number,
            scene_title=current_scene.title,
            location=location,  # Use location from current instruction
            total_scenes=len(story.scenes),
        )

        # Get player character
        player_character = next(
            (char.name for char in story.characters if char.is_main_character),
            None,
        )

        # Build NPC list (exclude main character)
        available_npcs = [
            CharacterInfo(
                character_id=char.character_id or char.name.lower(),
                name=char.name,
                age=char.age,
                sex=char.sex,
                model=char.model,
                role=char.name.lower(),  # Use name as role
            )
            for char in story.characters
            if not char.is_main_character
        ]

        # Get next instruction
        next_instr_dict = self._get_next_instruction(story, session.state)
        logger.info(f"🔍 DEBUG next_instr_dict keys: {list(next_instr_dict.keys()) if next_instr_dict else None}")
        logger.info(f"🔍 DEBUG next_instr_dict['choices']: {next_instr_dict.get('choices') if next_instr_dict else None}")
        next_instruction = NextInstructionInfo(**next_instr_dict) if next_instr_dict else None

        # Get progress info
        progress_dict = self._get_current_beat_info(story, session.state)
        progress = ProgressInfo(**progress_dict)

        # Build metadata
        metadata = {
            "last_updated": session.updated_at.isoformat() if session.updated_at else None,
            "total_interactions": len(session.state.completed_dialogue_beats),
            "current_instruction_index": session.state.current_instruction_index,
        }

        # Build nested relationships object for response
        from letta.schemas.story import SessionStateRelationships
        
        relationships_defined = []
        if story.relationships:
            relationships_defined = [rel.relationship_id for rel in story.relationships if rel.relationship_id]
        
        relationships_obj = SessionStateRelationships(
            relationship_points=session.state.relationship_points,
            relationship_levels=session.state.relationship_levels,
            relationships_defined=relationships_defined
        )

        return SessionStateResponse(
            story_id=story.story_id,
            story_title=story.title,
            session_id=session.session_id,
            session_status=session.status.value,
            current_setting=current_setting,
            player_character=player_character,
            available_npcs=available_npcs,
            next_instruction=next_instruction,
            progress=progress,
            state=session.state,  # Q7 FIX: Include raw state for completion tracking
            relationships=relationships_obj,  # NEW: Nested relationships object
            metadata=metadata,
        )

    async def get_story_details(
        self,
        story_id: str,
        actor: User,
    ):
        """
        Get full story structure for caching in Unity.

        Returns complete story with all scenes, characters, and instructions.
        This allows Kon's server to cache the story data locally.

        Args:
            story_id: Story identifier
            actor: User making the request

        Returns:
            StoryDetailResponse with full story structure
        """
        from letta.schemas.story import StoryDetailResponse

        # Get story
        story = await self.story_manager.get_story(story_id, actor)
        if not story:
            raise ValueError(f"Story '{story_id}' not found")

        # Separate player character and NPCs
        player_character = next(
            (char for char in story.characters if char.is_main_character),
            None,
        )

        npcs = [char for char in story.characters if not char.is_main_character]

        return StoryDetailResponse(
            story_id=story.story_id,
            title=story.title,
            description=story.description,
            characters=story.characters,
            player_character=player_character,
            npcs=npcs,
            scenes=story.scenes,
            total_scenes=len(story.scenes),
            metadata=story.metadata,
        )

    async def update_relationships_memory(
        self,
        session_id: str,
        story: "Story",
        session_state: SessionState,
        character_agents: Dict[str, str],
        actor: User,
    ) -> None:
        """
        Update relationship memory blocks for all character agents.
        
        Called after relationship effects are applied to keep AI agents' memory
        synchronized with actual relationship status.
        
        Args:
            session_id: Session identifier (for logging)
            story: Story with character definitions
            session_state: Current session state with updated relationships
            character_agents: Dict mapping character name to agent_id
            actor: User making the request
        """
        if not story.relationships:
            logger.debug(f"  ℹ️ No relationships to update (session: {session_id})")
            return
        
        logger.info(f"  🔄 Updating relationship memory for {len(character_agents)} agents...")
        
        from letta.schemas.block import BlockUpdate
        import asyncio
        
        async def update_agent_relationships(char_id: str, agent_id: str, character: StoryCharacter):
            """Update single agent's relationship memory block."""
            try:
                # Get agent with memory blocks
                agent = await self.agent_manager.get_agent_by_id_async(
                    agent_id=agent_id,
                    actor=actor,
                    include_relationships=["memory"]
                )
                
                # Find relationships memory block
                relationships_block = None
                for block in agent.memory.blocks:
                    if block.label == "relationships":
                        relationships_block = block
                        break
                
                if not relationships_block:
                    logger.warning(f"    ⚠️ No relationships block for {char_id} (agent {agent_id})")
                    return
                
                # Build updated relationships context
                updated_value = self._build_relationships_context(story, session_state, character)
                
                # Update block
                await self.block_manager.update_block_async(
                    block_id=relationships_block.id,
                    block_update=BlockUpdate(value=updated_value),
                    actor=actor,
                )
                
                logger.debug(f"    ✅ Updated relationships for {char_id}")
                
            except Exception as e:
                logger.error(f"    ❌ Failed to update relationships for {char_id}: {e}")
        
        # Create character lookup for agent updates (using character.name as key)
        char_lookup = {char.name: char for char in story.characters if char.name}
        
        # Update all agents in parallel
        update_tasks = [
            update_agent_relationships(char_id, agent_id, char_lookup[char_id])
            for char_id, agent_id in character_agents.items()
            if char_id in char_lookup  # Only update if we have character def
        ]
        
        await asyncio.gather(*update_tasks)
        logger.info(f"  ✅ Relationship memory updated for all agents")

    def apply_relationship_effects(
        self,
        session_state: SessionState,
        story: "Story",
        choice_id: int,
        current_instruction: Dict[str, Any],
    ) -> None:
        """
        Apply relationship effects from a player choice.
        
        Modifies session_state in place by updating relationship_points and relationship_levels.
        
        NOTE: After calling this, you MUST call update_relationships_memory() to sync
        the AI agents' memory blocks with the new relationship values.
        
        Args:
            session_state: Current session state (will be modified)
            story: Story with relationship definitions
            choice_id: ID of the choice that was selected
            current_instruction: Current instruction dict containing choices
            
        Raises:
            ValueError: If choice not found or invalid relationship reference
        """
        if not story.relationships:
            logger.debug("  ℹ️ Story has no relationships defined, skipping effect application")
            return
        
        # Extract choices from instruction
        choices = current_instruction.get("choices")
        if not choices:
            logger.debug("  ℹ️ Current instruction has no choices")
            return
        
        # Find the selected choice
        from letta.schemas.story import StoryChoice
        
        selected_choice = None
        for choice_data in choices:
            # Handle both dict and StoryChoice objects
            if isinstance(choice_data, dict):
                if choice_data.get("id") == choice_id:
                    # Parse as StoryChoice to get relationship_effects
                    selected_choice = StoryChoice(**choice_data)
                    break
            elif isinstance(choice_data, StoryChoice):
                if choice_data.id == choice_id:
                    selected_choice = choice_data
                    break
        
        if not selected_choice:
            logger.warning(f"  ⚠️ Choice {choice_id} not found in current instruction")
            return
        
        if not selected_choice.relationship_effects:
            logger.debug(f"  ℹ️ Choice {choice_id} has no relationship effects")
            return
        
        # Build relationship lookup: (character, type) -> relationship_def
        rel_lookup = {(rel.character, rel.type): rel for rel in story.relationships}
        
        # Apply each effect
        for effect in selected_choice.relationship_effects:
            # Find relationship definition
            rel_def = rel_lookup.get((effect.character, effect.type))
            if not rel_def:
                logger.warning(
                    f"  ⚠️ Unknown relationship in effect: character='{effect.character}', type='{effect.type}'"
                )
                continue
            
            rel_id = rel_def.relationship_id
            
            # Parse effect string ('+10', '-5', etc.)
            try:
                change = int(effect.effect)
            except ValueError:
                logger.warning(f"  ⚠️ Invalid effect value: '{effect.effect}' (expected '+10' or '-5')")
                continue
            
            # Update points (using array helper methods)
            current_points = self._find_relationship_point(session_state.relationship_points, rel_id)
            if current_points is None:
                current_points = rel_def.starting_points
            
            max_points = rel_def.max_levels * rel_def.points_per_level  # Calculate maximum points
            new_points = max(0, min(max_points, current_points + change))  # Floor at 0, cap at max
            
            self._update_relationship_point(session_state.relationship_points, rel_id, new_points)
            
            # Recalculate level (1-based)
            if rel_def.points_per_level > 0:
                new_level = (new_points // rel_def.points_per_level) + 1
                new_level = min(new_level, rel_def.max_levels)  # Cap at max
                new_level = max(new_level, 1)  # Floor at 1
            else:
                new_level = 1
            
            old_level = self._find_relationship_level(session_state.relationship_levels, rel_id) or 1
            self._update_relationship_level(session_state.relationship_levels, rel_id, new_level)
            
            # Log the change
            level_change = ""
            if new_level != old_level:
                level_change = f" (level {old_level} → {new_level})"
            
            logger.info(
                f"  💝 Relationship '{rel_id}': {current_points} → {new_points} points{level_change}"
        )

    async def update_session_state_with_version(
        self,
        session_id: str,
        state: SessionState,
        expected_version: int,
        actor: User,
    ) -> Dict[str, bool | int]:
        """
        Update session state with optimistic locking to prevent race conditions.
        
        Args:
            session_id: Session to update
            state: New state to save
            expected_version: Expected current version (for optimistic lock)
            actor: User performing the update
            
        Returns:
            Dict with 'success': bool and either 'new_version': int or 'current_version': int
        """
        from sqlalchemy import update
        
        state_dict = state.model_dump(mode="json") if hasattr(state, "model_dump") else state.dict()
        
        async with db_registry.async_session() as db_session:
            async with db_session.begin():
                # Update only if version matches (optimistic lock)
                stmt = (
                    update(StorySessionORM)
                    .where(
                        StorySessionORM.session_id == session_id,
                        StorySessionORM.version == expected_version
                    )
                    .values(
                        state=state_dict,
                        version=expected_version + 1,
                        updated_at=datetime.utcnow()
                    )
                )
                result = await db_session.execute(stmt)
                
                if result.rowcount == 0:
                    # Version mismatch - state was modified by another request
                    check_stmt = select(StorySessionORM.version).where(
                        StorySessionORM.session_id == session_id
                    )
                    current_version_result = await db_session.execute(check_stmt)
                    current_version_row = current_version_result.first()
                    current_version = current_version_row[0] if current_version_row else None
                    
                    logger.warning(
                        f"  ⚠️ Version mismatch for session {session_id}: "
                        f"expected={expected_version}, current={current_version}"
                    )
                    return {
                        "success": False,
                        "current_version": current_version,
                    }
                else:
                    logger.debug(
                        f"  ✅ Session state updated: {session_id} "
                        f"(version: {expected_version} → {expected_version + 1})"
                    )
                    return {
                        "success": True,
                        "new_version": expected_version + 1,
                    }

    # ============================================================
    # Core Memory Updates - Scene Progression
    # ============================================================

    async def update_scene_memory_blocks(
        self,
        session_id: str,
        new_scene_number: int,
        actor: User,
    ) -> Dict[str, any]:
        """
        Update all character agents' current_scene memory block when scene changes.
        
        This ensures NPCs always reference the correct scene context (location, mood, etc.)
        instead of being stuck in Scene 1 throughout the entire story.
        
        **What Gets Updated:**
        - Current scene info (scene number, title, location, mood)
        - Scene history (tracks journey: "we were in factory, now on rooftop")
        
        **Called From:**
        1. dialogue_manager.py - After auto-advance from dialogue beats
        2. story.py /advance-story - After auto-advance from narration/action
        3. story.py /advance-scene - After manual scene advance
        
        **Performance:**
        - Updates run in parallel using asyncio.gather()
        - Robust error handling (one failure doesn't block others)
        
        Args:
            session_id: Story session ID
            new_scene_number: New scene number (1-indexed)
            actor: User performing the action
            
        Returns:
            Dict with success status and update counts
            
        Raises:
            ValueError: If session or story not found
            
        Example:
            >>> await session_manager.update_scene_memory_blocks(
            ...     session_id="session-123",
            ...     new_scene_number=5,
            ...     actor=user
            ... )
            {
                "success": True,
                "updated_agents": 3,
                "failed_agents": 0,
                "total_agents": 3
            }
        """
        logger.info(
            f"🔄 Updating scene memory blocks: "
            f"session={session_id}, new_scene={new_scene_number}"
        )
        
        # Step 1: Get session and story
        session = await self._get_session_by_id(session_id, actor)
        if not session:
            raise ValueError(f"Session '{session_id}' not found")
        
        story = await self.story_manager.get_story(session.story_id, actor)
        if not story:
            raise ValueError(f"Story '{session.story_id}' not found")
        
        # Step 2: Validate scene number
        if new_scene_number < 1 or new_scene_number > len(story.scenes):
            logger.warning(
                f"  ⚠️ Invalid scene number: {new_scene_number} "
                f"(valid range: 1-{len(story.scenes)})"
            )
            return {
                "success": False,
                "error": "Invalid scene number",
                "updated_agents": 0,
                "failed_agents": 0,
                "total_agents": len(session.character_agents),
            }
        
        # Step 3: Build new scene context with history
        new_scene = story.scenes[new_scene_number - 1]
        scene_value = self._build_scene_context_with_history(
            current_scene=new_scene,
            all_scenes=story.scenes,
            current_scene_number=new_scene_number,
        )
        
        # Step 4: Update all character agents in parallel
        from letta.schemas.block import BlockUpdate
        import asyncio
        
        async def update_agent_scene_memory(char_id: str, agent_id: str) -> Dict[str, any]:
            """Update single agent's scene memory. Returns result dict."""
            try:
                # Get agent to find current_scene block ID
                agent = await self.agent_manager.get_agent_by_id_async(
                    agent_id=agent_id,
                    actor=actor,
                    include_relationships=["memory"]  # CRITICAL: Load memory blocks!
                )
                
                # Find current_scene block
                scene_block = None
                for block in agent.memory.blocks:
                    if block.label == "current_scene":
                        scene_block = block
                        break
                
                if not scene_block:
                    logger.warning(
                        f"  ⚠️ No current_scene block found for {char_id} "
                        f"(agent {agent_id})"
                    )
                    return {
                        "char_id": char_id,
                        "agent_id": agent_id,
                        "success": False,
                        "error": "No current_scene block found",
                    }
                
                # Update block value
                await self.block_manager.update_block_async(
                    block_id=scene_block.id,
                    block_update=BlockUpdate(value=scene_value),
                    actor=actor,
                )
                
                logger.debug(f"  ✅ Updated scene memory: {char_id} (agent {agent_id})")
                return {
                    "char_id": char_id,
                    "agent_id": agent_id,
                    "success": True,
                }
                
            except Exception as e:
                logger.error(
                    f"  ❌ Failed to update scene memory for {char_id}: {e}",
                    exc_info=True,
                )
                return {
                    "char_id": char_id,
                    "agent_id": agent_id,
                    "success": False,
                    "error": str(e),
                }
        
        # Execute updates in parallel (performance optimization)
        update_tasks = [
            update_agent_scene_memory(char_id, agent_id)
            for char_id, agent_id in session.character_agents.items()
        ]
        
        results = await asyncio.gather(*update_tasks, return_exceptions=False)
        
        # Step 5: Count successes and failures
        updated_count = sum(1 for r in results if r.get("success"))
        failed_count = sum(1 for r in results if not r.get("success"))
        total_count = len(results)
        
        # Step 6: Log summary
        if failed_count == 0:
            logger.info(
                f"✅ Successfully updated scene memory for all {updated_count} agents "
                f"(Scene {new_scene_number}: {new_scene.title})"
            )
        else:
            logger.warning(
                f"⚠️ Updated {updated_count}/{total_count} agents "
                f"({failed_count} failed)"
            )
            for result in results:
                if not result.get("success"):
                    logger.warning(
                        f"  - {result['char_id']}: {result.get('error', 'Unknown error')}"
                    )
        
        return {
            "success": updated_count > 0,  # Success if at least one updated
            "updated_agents": updated_count,
            "failed_agents": failed_count,
            "total_agents": total_count,
            "scene_number": new_scene_number,
            "scene_title": new_scene.title,
        }
    
    def _build_scene_context_with_history(
        self,
        current_scene: Scene,
        all_scenes: List[Scene],
        current_scene_number: int,
    ) -> str:
        """
        Build scene context with history tracking.
        
        Instead of just current scene, this includes:
        1. Current scene details (location, mood)
        2. Scene history (where we've been)
        3. Journey context for better NPC memory
        
        Args:
            current_scene: Current scene object
            all_scenes: All scenes in the story
            current_scene_number: Current scene number (1-indexed)
            
        Returns:
            Formatted scene context string with history
            
        Example Output:
            === CURRENT SCENE ===
            Scene 5: The Final Confrontation
            Location: Corporate Headquarters - Rooftop
            
            SCENE HISTORY (Your Journey So Far):
            • Scene 1: The Awakening - Abandoned Factory
            • Scene 2: The System's Whispers - City Streets
            • Scene 3: The First Trial - Underground Lab
            • Scene 4: The Revelation - Corporate Headquarters - Lobby
            • Scene 5: The Final Confrontation - Corporate Headquarters - Rooftop (CURRENT)
            
            You are currently in this scene. Remember your journey and respond naturally.
        """
        context_parts = [
            "=== CURRENT SCENE ===",
            f"Scene {current_scene.scene_number}: {current_scene.title}",
            "",
        ]
        
        # Add location
        if current_scene.location:
            context_parts.append(f"Location: {current_scene.location}")
        
        # Extract mood from scene (if available)
        mood_indicators = []
        if "mood:" in current_scene.location.lower():
            # Extract mood from "Mood: desperate, fearful" format
            mood_part = current_scene.location.lower().split("mood:")[-1].strip()
            mood_indicators.append(f"Mood: {mood_part}")
        
        if mood_indicators:
            context_parts.extend(mood_indicators)
        
        # Add scene history (user requested this)
        if current_scene_number > 1:
            context_parts.extend([
                "",
                "SCENE HISTORY (Your Journey So Far):",
            ])
            
            # List all scenes up to current
            for scene in all_scenes[:current_scene_number]:
                scene_marker = " (CURRENT)" if scene.scene_number == current_scene_number else ""
                context_parts.append(
                    f"• Scene {scene.scene_number}: {scene.title} - "
                    f"{scene.location}{scene_marker}"
                )
        
        # Add context reminder
        context_parts.extend([
            "",
            "You are currently in this scene. Remember your journey and respond naturally to what happens.",
        ])
        
        return "\n".join(context_parts)
