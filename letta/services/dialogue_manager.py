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
from letta.services.semantic_evaluation_service import SemanticEvaluationService
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
        self.semantic_evaluator = SemanticEvaluationService()
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
            f"message='{request.player_message[:50]}...', "
            f"user={actor.id}"
        )
        
        try:
            # Step 1: Get session and story
            logger.debug(f"  📋 Fetching session: {session_id}")
            session = await self.session_manager._get_session_by_id(session_id, actor)
            if not session:
                logger.error(f"  ❌ Session not found: {session_id}")
                raise ValueError(f"Session '{session_id}' not found")
            
            logger.debug(f"  ✅ Session found, story_id: {session.story_id}")
            
            # Backwards compat: initialize new fields if missing
            if not hasattr(session.state, 'dialogue_attempts'):
                session.state.dialogue_attempts = {}
            if not hasattr(session.state, 'semantic_similarity_scores'):
                session.state.semantic_similarity_scores = {}
            
            story = await self.story_manager.get_story(session.story_id, actor)
            if not story:
                raise ValueError(f"Story '{session.story_id}' not found")
            
            # NEW: Check if semantic validation is enabled for this story
            use_semantic_validation = story.metadata.get(
                "scene_progression_settings", {}
            ).get("use_semantic_validation", False)
            max_attempts = story.metadata.get(
                "scene_progression_settings", {}
            ).get("max_off_topic_attempts", 4)
            
            logger.debug(f"  🧠 Semantic validation: {use_semantic_validation}, max attempts: {max_attempts}")
            
            # 🎮 FIRST-PERSON MODE: Validate player isn't talking to themselves
            # Find main character
            main_character = next(
                (char for char in story.characters if char.is_main_character),
                None
            )
            
            # Block if player tries to talk to main character (case-insensitive)
            if main_character:
                target_char_lower = request.target_character.lower()
                main_char_id_lower = (main_character.character_id or "").lower()
                main_char_name_lower = main_character.name.lower()
                
                if target_char_lower in [main_char_id_lower, main_char_name_lower]:
                    logger.warning(f"  ⚠️ Player tried to talk to themselves: '{request.target_character}'")
                    raise ValueError(
                        f"❌ Cannot talk to '{main_character.name}' - YOU ARE {main_character.name}! "
                        f"You're playing as the main character. Talk to OTHER characters instead. "
                        f"Available: {list(session.character_agents.keys())}"
                    )
            
            # Step 2: Get current scene
            current_scene_num = session.state.current_scene_number
            if current_scene_num > len(story.scenes):
                raise ValueError(f"Scene {current_scene_num} out of range")
            
            current_scene = story.scenes[current_scene_num - 1]  # 1-indexed
            logger.debug(f"  📍 Current scene: {current_scene.title}")
            
            # Q8: Character Presence Validation - Ensure character is in current scene
            # Convert to lowercase for case-insensitive matching
            target_character_lower = request.target_character.lower()
            if target_character_lower not in current_scene.characters:
                # Get character names for better error message
                present_character_names = []
                for char_id in current_scene.characters:
                    char = next((c for c in story.characters if c.character_id == char_id), None)
                    if char:
                        present_character_names.append(f"{char.name} ({char_id})")
                
                # Find target character name
                target_char_obj = next(
                    (c for c in story.characters if c.character_id == target_character_lower),
                    None
                )
                target_name = target_char_obj.name if target_char_obj else request.target_character
                
                raise ValueError(
                    f"❌ Character '{target_name}' is not present in the current scene! "
                    f"Current scene: '{current_scene.title}' (Scene {current_scene_num}). "
                    f"Characters present: {', '.join(present_character_names) if present_character_names else 'None'}. "
                    f"You can only talk to characters who are in the current scene."
                )
            
            # Step 3: Get character agent (case-insensitive lookup)
            logger.debug(f"  👤 Looking up character: '{request.target_character}'")
            logger.debug(f"  📋 Available agents in session: {list(session.character_agents.keys())}")
            
            # Find character agent (case-insensitive)
            agent_id = None
            matched_char_name = None
            for char_name, char_agent_id in session.character_agents.items():
                if char_name.lower() == target_character_lower:
                    agent_id = char_agent_id
                    matched_char_name = char_name
                    break
            
            if not agent_id:
                logger.error(f"  ❌ Character '{request.target_character}' not found!")
                logger.error(f"  📋 Available characters: {list(session.character_agents.keys())}")
                raise ValueError(
                    f"Character '{request.target_character}' not found in session. "
                    f"Available: {list(session.character_agents.keys())}"
                )
            
            logger.debug(f"  ✅ Found agent: {agent_id} for character '{matched_char_name}'")
            
            # Step 4: Get next dialogue beat for this character (use lowercase)
            next_beat = self._get_next_dialogue_beat(
                current_scene,
                target_character_lower,
                session.state,
                story.characters,
            )
            
            # Step 5: Generate dialogue with script guidance (use lowercase)
            character_response, emotion, animation = await self._generate_with_script_guidance(
                agent_id=agent_id,
                player_message=request.player_message,
                character_name=target_character_lower,
                next_beat=next_beat,
                scene=current_scene,
                actor=actor,
            )
            
            # Step 6: Mark beat as completed and auto-advance
            beats_completed = []
            if next_beat:
                beat_id = next_beat["beat_id"]
                
                # NEW: Initialize attempt tracking if not exists
                if beat_id not in session.state.dialogue_attempts:
                    session.state.dialogue_attempts[beat_id] = 0
                
                # Increment attempt counter
                session.state.dialogue_attempts[beat_id] += 1
                current_attempts = session.state.dialogue_attempts[beat_id]
                
                logger.info(f"  📊 Beat {beat_id}: Attempt {current_attempts}/{max_attempts}")
                
                # Determine if beat should be marked complete
                should_complete_beat = False
                
                if not use_semantic_validation:
                    # Legacy behavior: always complete (backwards compatible)
                    should_complete_beat = True
                    logger.debug(f"  ⚡ Auto-advance (semantic validation disabled)")
                    
                elif current_attempts >= max_attempts:
                    # Max attempts reached: auto-complete
                    should_complete_beat = True
                    logger.info(f"  ⏭️ Auto-complete after {max_attempts} attempts")
                    
                else:
                    # Evaluate with LLM
                    logger.debug(f"  🧠 Evaluating semantic completion...")
                    beat_objective = next_beat.get("topic", next_beat.get("script_text", ""))
                    
                    was_addressed, confidence_score = await self.semantic_evaluator.evaluate_beat_completion(
                        beat_objective=beat_objective,
                        player_message=request.player_message,
                        character_response=character_response,
                        actor=actor,
                    )
                    
                    # Store similarity score
                    session.state.semantic_similarity_scores[beat_id] = confidence_score
                    
                    if was_addressed:
                        should_complete_beat = True
                        logger.info(f"  ✅ Beat addressed (confidence: {confidence_score:.2f})")
                    else:
                        logger.info(f"  ❌ Beat not addressed (confidence: {confidence_score:.2f})")
                
                # NOW mark beat as complete if determined
                if should_complete_beat:
                    beats_completed.append(beat_id)
                    session.state.completed_dialogue_beats.append(beat_id)
                    logger.info(f"  ✅ Beat completed: {beat_id}")
                    
                    # Reset attempt counter
                    session.state.dialogue_attempts[beat_id] = 0
                    
                    # Auto-advance instruction index
                    session.state.current_instruction_index += 1
                    logger.info(f"  ➡️ Auto-advancing to next instruction")
                else:
                    # Beat NOT complete - stay on same beat
                    logger.info(f"  🔄 Staying on beat {beat_id} (attempt {current_attempts}/{max_attempts})")
            
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
            
                # Update NPC memory blocks with new scene context
                try:
                    update_result = await self.session_manager.update_scene_memory_blocks(
                        session_id=session_id,
                        new_scene_number=next_scene_number,
                        actor=actor,
                    )
                    logger.info(
                        f"  🧠 Updated {update_result['updated_agents']}/{update_result['total_agents']} "
                        f"agent memories for Scene {next_scene_number}"
                    )
                except Exception as e:
                    # Don't fail the entire scene transition if memory update fails
                    logger.error(f"  ⚠️ Failed to update scene memories: {e}", exc_info=True)
            
            # Step 10: Return response
            character_full_name = self._get_character_name(
                target_character_lower,
                story.characters,
            )
            
            response = StoryDialogueResponse(
                character_id=target_character_lower,
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
            "TONE & RELATIONSHIP AWARENESS:",
            "- Natural and conversational",
            "- Stay in character",
            "- **IMPORTANT: Check your RELATIONSHIP STATUS memory block**",
            "- Adjust your tone based on your relationship level with the player",
            "- Higher friendship = warmer, more familiar, trusting",
            "- Higher romance = more caring, flirty, intimate",
            "- Lower levels = more formal, distant, cautious",
            "- Don't break the fourth wall",
            "- Don't mention 'the script', 'the story', or 'relationship points'",
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
        
        Q4: Now respects beat priority!
        - Only "required" beats block scene completion
        - "optional" beats don't prevent scene transition
        - Progress is calculated over ALL beats (required + optional)
        
        Returns:
            (is_complete, progress_fraction)
        """
        if not scene.dialogue_beats:
            return True, 1.0
        
        # Q4: Separate required and optional beats
        required_beats = [
            beat for beat in scene.dialogue_beats
            if beat.get("priority", "required") == "required"
        ]
        optional_beats = [
            beat for beat in scene.dialogue_beats
            if beat.get("priority", "required") == "optional"
        ]
        
        # Count completed beats (both required and optional)
        total_beats = len(scene.dialogue_beats)
        completed_count = sum(
            1 for beat in scene.dialogue_beats
            if beat.get("beat_id") in completed_beats
        )
        
        # Scene is complete if all REQUIRED beats are done
        required_completed = sum(
            1 for beat in required_beats
            if beat.get("beat_id") in completed_beats
        )
        is_complete = required_completed >= len(required_beats) if required_beats else True
        
        # Progress is over ALL beats (gives optional beats value)
        progress = completed_count / total_beats if total_beats > 0 else 1.0
        
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

