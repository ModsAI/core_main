"""
Dialogue Manager Service

Handles dialogue generation with script guidance.
Implements natural conversation while progressing the story.
"""

import re
from typing import Dict, List, Optional, Tuple

from letta.log import get_logger
from letta.schemas.agent import AgentType
from letta.schemas.letta_message import LettaMessage
from letta.schemas.message import MessageCreate, MessageRole
from letta.schemas.story import (
    DialogueBeatInfo,
    Scene,
    SessionState,
    StoryDialogueRequest,
    StoryDialogueResponse,
    StorySession,
)
from letta.schemas.user import User
from letta.server.db import db_registry
from letta.services.agent_manager import AgentManager
from letta.services.session_manager import SessionManager
from letta.services.story_manager import StoryManager

logger = get_logger(__name__)


class DialogueManager:
    """
    Manages dialogue generation with script guidance.
    
    Key responsibilities:
    1. Get next dialogue beat from script
    2. Build context-aware system prompt
    3. Send player message to character agent
    4. Parse agent response
    5. Mark beats as completed
    6. Auto-advance scenes when beats done
    7. Handle errors with fallback
    """

    def __init__(self, server=None):
        self.server = server  # Will be injected from router
        self.agent_manager = AgentManager()
        self.session_manager = SessionManager()
        self.story_manager = StoryManager()
        logger.info("💬 DialogueManager initialized")

    async def generate_dialogue(
        self,
        session_id: str,
        request: StoryDialogueRequest,
        actor: User,
    ) -> StoryDialogueResponse:
        """
        Generate character dialogue with script guidance.
        
        **User Requirements:**
        - Q6: Use script as CONTEXT, respond naturally, guide conversation forward
        - Q7: Answer off-topic briefly, then redirect to script
        - Q8: Auto-advance scene when all beats completed
        - Q9: No scene skipping (must complete all beats)
        - Q10: If agent fails, use scripted text + retry
        
        Args:
            session_id: Story session
            request: Player message + target character
            actor: User playing
            
        Returns:
            Character response with script progress
        """
        logger.info(
            f"💬 Dialogue request: session={session_id}, "
            f"character={request.target_character}, "
            f"message='{request.player_message[:50]}...'"
        )
        
        try:
            # Step 1: Get session and story
            session = await self.session_manager._get_session_by_id(session_id, actor)
            if not session:
                raise ValueError(f"Session '{session_id}' not found")
            
            story = await self.story_manager.get_story(session.story_id, actor)
            if not story:
                raise ValueError(f"Story '{session.story_id}' not found")
            
            # Step 2: Get current scene
            current_scene_num = session.state.current_scene_number
            if current_scene_num > len(story.scenes):
                raise ValueError(f"Scene {current_scene_num} out of range")
            
            current_scene = story.scenes[current_scene_num - 1]  # 1-indexed
            logger.debug(f"  📍 Current scene: {current_scene.title}")
            
            # Step 3: Get character agent
            agent_id = session.character_agents.get(request.target_character)
            if not agent_id:
                raise ValueError(
                    f"Character '{request.target_character}' not found in session. "
                    f"Available: {list(session.character_agents.keys())}"
                )
            
            # Step 4: Get next dialogue beat for this character
            next_beat = self._get_next_dialogue_beat(
                current_scene,
                request.target_character,
                session.state,
                story.characters,
            )
            
            # Step 5: Generate dialogue with script guidance
            character_response, emotion, animation = await self._generate_with_script_guidance(
                agent_id=agent_id,
                player_message=request.player_message,
                character_name=request.target_character,
                next_beat=next_beat,
                scene=current_scene,
                actor=actor,
            )
            
            # Step 6: Check if beat was completed
            beats_completed = []
            if next_beat and self._beat_was_addressed(character_response, next_beat):
                beats_completed.append(next_beat["beat_id"])
                session.state.completed_dialogue_beats.append(next_beat["beat_id"])
                logger.info(f"  ✅ Beat completed: {next_beat['beat_id']}")
            
            # Step 7: Update session state
            await self._update_session_state(session_id, session.state, actor)
            
            # Step 8: Check scene completion
            scene_complete, scene_progress = self._check_scene_completion(
                current_scene,
                session.state.completed_dialogue_beats,
            )
            
            # Step 9: Auto-advance to next scene if complete (Q8: Option A)
            next_scene_number = None
            if scene_complete and current_scene_num < len(story.scenes):
                next_scene_number = current_scene_num + 1
                session.state.current_scene_number = next_scene_number
                session.state.current_instruction_index = 0
                session.state.completed_dialogue_beats = []  # Reset for new scene
                await self._update_session_state(session_id, session.state, actor)
                logger.info(f"  🎬 Scene complete! Auto-advancing to scene {next_scene_number}")
            
            # Step 10: Return response
            character_full_name = self._get_character_name(
                request.target_character,
                story.characters,
            )
            
            response = StoryDialogueResponse(
                character_id=request.target_character,
                character_name=character_full_name,
                dialogue_text=character_response,
                emotion=emotion,
                animation_suggestion=animation,
                dialogue_beats_completed=beats_completed,
                scene_progress=scene_progress,
                scene_complete=scene_complete,
                session_updated=True,
                next_scene_number=next_scene_number,
            )
            
            logger.info(
                f"  ✅ Dialogue generated: {len(character_response)} chars, "
                f"beats={len(beats_completed)}, progress={scene_progress:.1%}, "
                f"complete={scene_complete}"
            )
            
            return response
        
        except Exception as e:
            logger.error(f"❌ Dialogue generation failed: {e}", exc_info=True)
            raise

    async def _generate_with_script_guidance(
        self,
        agent_id: str,
        player_message: str,
        character_name: str,
        next_beat: Optional[Dict],
        scene: Scene,
        actor: User,
    ) -> Tuple[str, str, str]:
        """
        Generate dialogue with script guidance.
        
        Uses system prompt to tell agent:
        - Respond naturally to player (Q6: Option C)
        - If off-topic, answer briefly then redirect (Q7: Option B)
        - Work in the next script topic/dialogue
        - Keep conversation moving forward
        
        Fallback (Q10): If agent fails, use scripted text directly
        """
        try:
            # Build context message with script guidance
            context_message = self._build_script_guidance_prompt(
                character_name=character_name,
                next_beat=next_beat,
                scene=scene,
                player_message=player_message,
            )
            
            logger.debug(f"  📝 Script guidance prompt: {context_message[:200]}...")
            
            # Send message to agent
            message_create = MessageCreate(
                role=MessageRole.user,
                content=f"{context_message}\n\nPlayer: {player_message}",  # Fixed: was 'text', should be 'content'
            )
            
            # Try to send message (with retry on failure - Q10: Option C)
            max_retries = 3
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    logger.debug(f"  🔄 Attempt {attempt + 1}/{max_retries}")
                    
                    # Use send_messages with StreamingServerInterface (correct for Letta 0.8.9)
                    from letta.server.rest_api.interface import StreamingServerInterface
                    
                    # Create interface to capture messages
                    interface = StreamingServerInterface()
                    
                    # Call send_messages (synchronous, even in async context)
                    usage_stats = self.server.send_messages(
                        actor=actor,
                        agent_id=agent_id,
                        input_messages=[message_create],
                        interface=interface,
                    )
                    
                    # Get recent messages from the agent's message history
                    # Use message_manager directly (correct for Letta 0.8.9)
                    from letta.services.message_manager import MessageManager
                    
                    message_manager = MessageManager()
                    recent_messages = await message_manager.list_messages_for_agent_async(
                        agent_id=agent_id,
                        actor=actor,
                        limit=5,  # Get last 5 messages (should include our response)
                        ascending=False,  # Get newest first
                    )
                    
                    logger.debug(f"  📦 Received {len(recent_messages)} messages from message history")
                    
                    # Extract character response from recent messages
                    character_response = self._extract_character_dialogue(recent_messages)
                    
                    if character_response:
                        # Parse emotion and animation from response
                        emotion, animation = self._parse_emotion_and_animation(
                            character_response
                        )
                        
                        return character_response, emotion, animation
                    
                    # If no response, try again
                    last_error = "No dialogue in agent response"
                    logger.warning(f"  ⚠️ {last_error}, retrying...")
                    continue
                
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"  ⚠️ Agent error: {e}, retrying...")
                    import traceback
                    logger.debug(f"  📜 Full traceback: {traceback.format_exc()}")
                    continue
            
            # All retries failed - use fallback (Q10: scripted text)
            logger.error(f"  ❌ All retries failed: {last_error}")
            logger.info(f"  🔄 Fallback: Using scripted text")
            
            if next_beat and next_beat.get("script_text"):
                return (
                    next_beat["script_text"],
                    "neutral",
                    "idle",
                )
            else:
                return (
                    f"[{character_name}] I'm not sure what to say right now. Let's continue the story.",
                    "neutral",
                    "idle",
                )
        
        except Exception as e:
            logger.error(f"  ❌ Fatal dialogue error: {e}", exc_info=True)
            # Ultimate fallback
            return (
                f"[{character_name}] ...",
                "neutral",
                "idle",
            )

    def _build_script_guidance_prompt(
        self,
        character_name: str,
        next_beat: Optional[Dict],
        scene: Scene,
        player_message: str,
    ) -> str:
        """
        Build system prompt with script guidance.
        
        Requirements:
        - Q6: Use script as context, respond naturally
        - Q7: Answer off-topic briefly, then redirect
        """
        prompt_parts = [
            f"You are {character_name} in the story scene: {scene.title}",
            f"Location: {scene.location}",
            "",
            "IMPORTANT INSTRUCTIONS:",
        ]
        
        if next_beat:
            prompt_parts.extend([
                f"📜 Script Context: The story wants you to talk about: {next_beat.get('topic', 'the script')}",
                f"Script Reference: \"{next_beat.get('script_text', '')}\"",
                "",
                "HOW TO RESPOND:",
                "1. First, respond naturally to what the player said",
                "2. Keep your response conversational and in-character",
                "3. Then, smoothly guide the conversation toward the script topic above",
                "4. Work in the script topic naturally - don't force it",
                "",
                "If the player asks something off-topic:",
                "- Answer their question BRIEFLY (1 sentence)",
                "- Then redirect to the script topic",
                f"- Example: 'Yeah, [brief answer]. But hey, {next_beat.get('topic', 'let me tell you something')}...'",
            ])
        else:
            prompt_parts.extend([
                "HOW TO RESPOND:",
                "1. Respond naturally to what the player said",
                "2. Stay in character",
                "3. Keep the conversation flowing",
            ])
        
        prompt_parts.extend([
            "",
            "TONE:",
            "- Natural and conversational",
            "- Stay in character",
            "- Don't break the fourth wall",
            "- Don't mention 'the script' or 'the story'",
            "",
            "Now, respond to the player:",
        ])
        
        return "\n".join(prompt_parts)

    def _extract_character_dialogue(self, messages: List[LettaMessage]) -> str:
        """Extract character dialogue from agent messages (from database)"""
        logger.debug(f"  📦 Extracting dialogue from {len(messages)} messages")
        
        # Look for assistant messages with tool_calls
        for i, message in enumerate(messages):
            logger.debug(f"  📨 Message {i}: role={getattr(message, 'role', 'unknown')}")
            
            # Only look at assistant messages
            if message.role != 'assistant':
                continue
            
            # Check for tool_calls (new Letta format)
            if hasattr(message, 'tool_calls') and message.tool_calls:
                logger.debug(f"    🔧 Found {len(message.tool_calls)} tool calls")
                for tool_call in message.tool_calls:
                    if tool_call.function.name == 'send_message':
                        import json
                        try:
                            args = json.loads(tool_call.function.arguments)
                            logger.debug(f"    📝 send_message arguments: {args}")
                            if 'message' in args:
                                dialogue = args['message']
                                logger.debug(f"    ✅ Extracted dialogue: {dialogue[:100]}...")
                                return dialogue
                        except Exception as e:
                            logger.debug(f"    ❌ Failed to parse tool call arguments: {e}")
            
            # Check for content (TextContent)
            if hasattr(message, 'content') and message.content:
                for content_item in message.content:
                    if hasattr(content_item, 'text') and content_item.text:
                        text = content_item.text
                        logger.debug(f"    💬 Found text content: {text[:100]}...")
                        # Skip inner thoughts
                        if not text.startswith('[') or not ']' in text:
                            return text
        
        logger.warning("  ⚠️ No dialogue extracted from messages!")
        return ""

    def _parse_emotion_and_animation(self, text: str) -> Tuple[str, str]:
        """
        Parse emotion and animation from dialogue text.
        
        Simple heuristic based on punctuation and keywords.
        """
        text_lower = text.lower()
        
        # Detect emotion
        if "!" in text or "wow" in text_lower or "great" in text_lower:
            emotion = "excited"
        elif "?" in text:
            emotion = "curious"
        elif "..." in text or "hmm" in text_lower:
            emotion = "thoughtful"
        elif any(word in text_lower for word in ["sad", "sorry", "unfortunately"]):
            emotion = "sad"
        elif any(word in text_lower for word in ["angry", "damn", "hate"]):
            emotion = "angry"
        else:
            emotion = "neutral"
        
        # Suggest animation based on emotion
        animation_map = {
            "excited": "gesture_enthusiastic",
            "curious": "lean_forward",
            "thoughtful": "think",
            "sad": "look_down",
            "angry": "cross_arms",
            "neutral": "idle",
        }
        
        animation = animation_map.get(emotion, "idle")
        
        return emotion, animation

    def _get_next_dialogue_beat(
        self,
        scene: Scene,
        character_id: str,
        state: SessionState,
        characters: List,
    ) -> Optional[Dict]:
        """
        Get next dialogue beat for this character in current scene.
        
        Returns None if all beats for this character are completed.
        """
        # Find character name from ID
        character_name = None
        for char in characters:
            if char.character_id == character_id:
                character_name = char.name
                break
        
        if not character_name:
            return None
        
        # Find next uncompleted beat for this character
        for beat in scene.dialogue_beats:
            if (
                beat.get("character") == character_name
                and beat.get("beat_id") not in state.completed_dialogue_beats
            ):
                return beat
        
        return None

    def _beat_was_addressed(self, response: str, beat: Dict) -> bool:
        """
        Check if dialogue beat was addressed in response.
        
        Simple keyword/topic detection.
        """
        response_lower = response.lower()
        
        # Get topic keywords
        topic = beat.get("topic", "").lower()
        script_text = beat.get("script_text", "").lower()
        
        # Extract keywords from topic (first 3 words)
        topic_keywords = topic.split()[:3]
        
        # Check if any topic keyword appears in response
        for keyword in topic_keywords:
            if len(keyword) > 3 and keyword in response_lower:
                return True
        
        # Check if script text appears (partial match)
        script_words = script_text.split()[:5]
        matches = sum(1 for word in script_words if len(word) > 3 and word in response_lower)
        
        if matches >= 2:  # At least 2 key words from script
            return True
        
        return False

    def _check_scene_completion(
        self,
        scene: Scene,
        completed_beats: List[str],
    ) -> Tuple[bool, float]:
        """
        Check if scene is complete.
        
        Returns:
            (is_complete, progress_fraction)
        """
        if not scene.dialogue_beats:
            return True, 1.0
        
        total_beats = len(scene.dialogue_beats)
        completed_count = sum(
            1 for beat in scene.dialogue_beats
            if beat.get("beat_id") in completed_beats
        )
        
        progress = completed_count / total_beats
        is_complete = completed_count >= total_beats
        
        return is_complete, progress

    def _get_character_name(self, character_id: str, characters: List) -> str:
        """Get character full name from ID"""
        for char in characters:
            if char.character_id == character_id:
                return char.name
        return character_id

    async def _update_session_state(
        self,
        session_id: str,
        state: SessionState,
        actor: User,
    ) -> None:
        """Update session state in database"""
        try:
            async with db_registry.async_session() as session:
                async with session.begin():
                    from sqlalchemy import update
                    from letta.orm.story import StorySession as StorySessionORM
                    
                    stmt = (
                        update(StorySessionORM)
                        .where(StorySessionORM.session_id == session_id)
                        .values(state=state.dict())
                    )
                    
                    await session.execute(stmt)
                    
                    logger.debug(f"  💾 Session state updated: {session_id}")
        
        except Exception as e:
            logger.error(f"  ❌ Failed to update session state: {e}")

