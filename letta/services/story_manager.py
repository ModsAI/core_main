"""
Story Manager Service

Handles story upload, parsing, and retrieval.
Converts TypeScript story JSON to our internal format.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from letta.log import get_logger
from letta.orm.story import Story as StoryORM
from letta.schemas.story import Scene, Story, StoryCharacter, StoryCharacterRelationship, StoryInstruction, StoryUpload, StoryUploadResponse
from letta.schemas.user import User
from letta.server.db import db_registry

logger = get_logger(__name__)


class StoryManager:
    """
    Manages story operations: upload, parse, retrieve, delete.
    
    Key responsibilities:
    1. Parse TypeScript story JSON
    2. Extract scenes from instructions
    3. Generate dialogue beats
    4. Store in database
    5. Handle errors gracefully
    """

    def __init__(self):
        logger.info("StoryManager initialized")

    async def upload_story(
        self,
        story_upload: StoryUpload,
        actor: User,
    ) -> StoryUploadResponse:
        """
        Upload and process a story.
        
        Process:
        1. Validate story structure
        2. Generate character IDs
        3. Parse instructions into scenes
        4. Extract dialogue beats
        5. Store in database
        
        Args:
            story_upload: Story data from TypeScript
            actor: User uploading the story
            
        Returns:
            Upload response with story details
            
        Raises:
            ValueError: Invalid story structure
            IntegrityError: Story ID already exists
            Exception: Other database/processing errors
        """
        logger.info(f" Uploading story: {story_upload.title} (ID: {story_upload.id})")
        
        try:
            # Step 1: Generate story_id
            story_id = f"story-{story_upload.id}"
            
            # Step 2: Process characters (add character_id)
            characters = self._process_characters(story_upload.characters)
            logger.debug(f"SUCCESS: Processed {len(characters)} characters")
            
            # Step 3: Process relationships (add relationship_id, validate)
            relationships = self._process_relationships(story_upload.relationships, characters)
            if relationships:
                logger.debug(f"SUCCESS: Processed {len(relationships)} relationships")
            
            # Step 4: Validate relationship effects in choices
            self._validate_relationship_effects(story_upload.instructions, relationships)
            
            # Step 5: Parse instructions into scenes
            scenes = self._parse_scenes(story_upload.instructions, characters)
            logger.debug(f"SUCCESS: Parsed {len(scenes)} scenes")
            
            # Step 6: Validate scene structure
            self._validate_scenes(scenes)
            
            # Step 5: Store in database
            async with db_registry.async_session() as session:
                async with session.begin():
                    # Check if story already exists
                    existing_check = select(StoryORM).where(StoryORM.story_id == story_id)
                    result = await session.execute(existing_check)
                    existing_story = result.scalar_one_or_none()
                    
                    if existing_story:
                        logger.error(f"ERROR: Story ID '{story_id}' already exists")
                        raise IntegrityError(f"Story ID '{story_id}' already exists", None, None)
                    
                    # Create ORM object
                    # Store processed story JSON with character_id and relationship_id populated
                    processed_story_json = story_upload.dict()
                    processed_story_json["characters"] = [char.dict() for char in characters]
                    processed_story_json["relationships"] = [rel.dict() for rel in relationships] if relationships else []
                    
                    # DEBUG: Check scenes before saving
                    scenes_dicts = [scene.dict() for scene in scenes]
                    for i, scene_dict in enumerate(scenes_dicts):
                        logger.info(f"DEBUG: Scene {i} has {len(scene_dict.get('narration_beats', []))} narration beats")
                        for j, beat in enumerate(scene_dict.get("narration_beats", [])):
                            logger.info(f"  DEBUG: Beat {j} has 'choices': {'choices' in beat}, value={beat.get('choices')}")

                    story_orm = StoryORM(
                        id=f"story-{uuid.uuid4()}",
                        story_id=story_id,
                        title=story_upload.title,
                        description=story_upload.description,
                        story_json=processed_story_json,  # Processed JSON with character_id and relationship_id
                        scenes_json={"scenes": scenes_dicts},  # Processed scenes
                        story_metadata={
                            "character_count": len(characters),
                            "relationship_count": len(relationships),
                            "scene_count": len(scenes),
                            "instruction_count": len(story_upload.instructions),
                            "tags": story_upload.tags or [],
                            "scene_progression_settings": story_upload.scene_progression_settings or {},
                        },
                        organization_id=actor.organization_id,
                    )
                    
                    session.add(story_orm)
                    await session.flush()
                    await session.refresh(story_orm)
                
                logger.info(f"SUCCESS: Successfully uploaded story: {story_upload.title}")
                
                return StoryUploadResponse(
                    success=True,
                    story_id=story_id,
                    title=story_upload.title,
                    scene_count=len(scenes),
                    character_count=len(characters),
                    instructions=[
                        f"Story '{story_upload.title}' uploaded successfully",
                        f"Story ID: {story_id}",
                        f"Scenes: {len(scenes)}",
                        f"Characters: {len(characters)}",
                        f"Start a session with: POST /api/v1/story/sessions/start",
                    ],
                )
        
        except IntegrityError as e:
            logger.error(f"ERROR: Story upload failed (duplicate): {e}")
            raise ValueError(f"Story with ID {story_upload.id} already exists") from e
        
        except ValueError as e:
            logger.error(f"ERROR: Story upload failed (validation): {e}")
            raise
        
        except SQLAlchemyError as e:
            logger.error(f"ERROR: Story upload failed (database): {e}", exc_info=True)
            raise Exception(f"Database error during story upload: {str(e)}") from e
        
        except Exception as e:
            logger.error(f"ERROR: Story upload failed (unexpected): {e}", exc_info=True)
            raise Exception(f"Failed to upload story: {str(e)}") from e

    async def get_story(
        self,
        story_id: str,
        actor: User,
    ) -> Optional[Story]:
        """
        Retrieve a story by ID.
        
        Args:
            story_id: Story identifier
            actor: User requesting the story
            
        Returns:
            Story object or None if not found
        """
        logger.debug(f"DEBUG: Getting story: {story_id}")
        
        try:
            async with db_registry.async_session() as session:
                async with session.begin():
                    query = select(StoryORM).where(
                        StoryORM.story_id == story_id,
                        StoryORM.organization_id == actor.organization_id,
                    )
                    result = await session.execute(query)
                    story_orm = result.scalar_one_or_none()
                    
                    if not story_orm:
                        logger.warning(f"ERROR: Story not found: {story_id}")
                        return None
                    
                    # Convert to schema
                    scenes = [Scene(**scene) for scene in story_orm.scenes_json["scenes"]]
                    characters = [StoryCharacter(**char) for char in story_orm.story_json["characters"]]
                    
                    # Parse relationships (if present)
                    relationships = None
                    if "relationships" in story_orm.story_json and story_orm.story_json["relationships"]:
                        relationships = [StoryCharacterRelationship(**rel) for rel in story_orm.story_json["relationships"]]
                    
                    story = Story(
                        story_id=story_orm.story_id,
                        title=story_orm.title,
                        description=story_orm.description,
                        characters=characters,
                        relationships=relationships,
                        scenes=scenes,
                        metadata=story_orm.story_metadata or {},
                    )
                    
                    logger.debug(f"SUCCESS: Found story: {story.title}")
                    return story
        
        except Exception as e:
            logger.error(f"ERROR: Failed to get story {story_id}: {e}", exc_info=True)
            return None

    async def delete_story(
        self,
        story_id: str,
        actor: User,
    ) -> bool:
        """
        Delete a story by ID.

        Args:
            story_id: Story identifier
            actor: User requesting the deletion

        Returns:
            True if deleted, False if not found or error
        """
        logger.info(f"Deleting story: {story_id}")

        try:
            from sqlalchemy import delete

            async with db_registry.async_session() as session:
                async with session.begin():
                    # Delete story
                    delete_stmt = delete(StoryORM).where(
                        StoryORM.story_id == story_id,
                        StoryORM.organization_id == actor.organization_id,
                    )
                    result = await session.execute(delete_stmt)

                    if result.rowcount > 0:
                        logger.info(f"SUCCESS: Story deleted: {story_id}")
                        return True
                    else:
                        logger.warning(f"ERROR: Story not found for deletion: {story_id}")
                        return False

        except Exception as e:
            logger.error(f"ERROR: Failed to delete story {story_id}: {e}", exc_info=True)
            return False

    async def list_stories(
        self,
        actor: User,
        page: int = 1,
        page_size: int = 50,
    ) -> Tuple[List[Dict], int]:
        """
        List all stories with metadata.
        
        Args:
            actor: User requesting the list
            page: Page number (1-indexed)
            page_size: Items per page
            
        Returns:
            Tuple of (story_list, total_count)
        """
        logger.debug(f"LISTING: Listing stories (page={page}, page_size={page_size})")
        
        try:
            async with db_registry.async_session() as session:
                async with session.begin():
                    # Count total stories for this organization
                    from sqlalchemy import func
                    count_query = select(func.count(StoryORM.id)).where(
                        StoryORM.organization_id == actor.organization_id,
                    )
                    count_result = await session.execute(count_query)
                    total = count_result.scalar() or 0
                    
                    # Query stories with pagination
                    offset = (page - 1) * page_size
                    query = (
                        select(StoryORM)
                        .where(StoryORM.organization_id == actor.organization_id)
                        .order_by(StoryORM.created_at.desc())
                        .limit(page_size)
                        .offset(offset)
                    )
                    result = await session.execute(query)
                    story_orms = result.scalars().all()
                    
                    # Format stories with metadata
                    stories = []
                    for story_orm in story_orms:
                        # Calculate counts from stored JSON
                        scenes_count = len(story_orm.scenes_json.get("scenes", []))
                        characters_count = len(story_orm.story_json.get("characters", []))
                        
                        # Estimate duration (5 minutes per scene as a heuristic)
                        estimated_minutes = scenes_count * 5
                        if estimated_minutes < 60:
                            estimated_duration = f"{estimated_minutes} min"
                        else:
                            hours = estimated_minutes // 60
                            remaining_minutes = estimated_minutes % 60
                            if remaining_minutes > 0:
                                estimated_duration = f"{hours}h {remaining_minutes}m"
                            else:
                                estimated_duration = f"{hours}h"
                        
                        story_item = {
                            "story_id": story_orm.story_id,
                            "title": story_orm.title,
                            "description": story_orm.description,
                            "scenes_count": scenes_count,
                            "characters_count": characters_count,
                            "estimated_duration": estimated_duration,
                            "created_at": story_orm.created_at,
                            "updated_at": story_orm.updated_at,
                            "metadata": story_orm.story_metadata or {},
                        }
                        stories.append(story_item)
                    
                    logger.info(f"SUCCESS: Found {len(stories)} stories (total: {total})")
                    return stories, total
        
        except Exception as e:
            logger.error(f"ERROR: Failed to list stories: {e}", exc_info=True)
            raise Exception(f"Failed to list stories: {str(e)}") from e

    def _process_characters(self, characters: List[StoryCharacter]) -> List[StoryCharacter]:
        """
        Process characters: generate character_id, validate.
        
        Args:
            characters: Raw characters from upload
            
        Returns:
            Processed characters with IDs
        """
        processed = []
        seen_ids = set()
        
        for char in characters:
            # Generate character_id from name (lowercase, replace spaces with underscores)
            char_id = char.name.lower().replace(" ", "_").replace("'", "")
            
            # Ensure uniqueness
            if char_id in seen_ids:
                char_id = f"{char_id}_{len(seen_ids)}"
            seen_ids.add(char_id)
            
            # Create new character with ID
            processed_char = StoryCharacter(
                name=char.name,
                sex=char.sex,
                age=char.age,
                is_main_character=char.is_main_character,
                model=char.model,
                character_id=char_id,
            )
            processed.append(processed_char)
            
            logger.debug(f"  CHARACTER: {char.name} -> {char_id}")
        
        return processed

    def _process_relationships(
        self, relationships: Optional[List[StoryCharacterRelationship]], characters: List[StoryCharacter]
    ) -> List[StoryCharacterRelationship]:
        """
        Process relationships: generate relationship_id, validate character references.

        Args:
            relationships: Raw relationships from upload
            characters: Processed characters (with character_id)

        Returns:
            Processed relationships with IDs
        """
        if not relationships:
            return []

        # Build character name -> character_id mapping
        char_name_to_id = {char.name: char.character_id for char in characters if char.character_id}

        processed = []
        seen_ids = set()

        for rel in relationships:
            # Validate character exists
            if rel.character not in char_name_to_id:
                raise ValueError(f"Relationship references unknown character: '{rel.character}'. Available characters: {list(char_name_to_id.keys())}")

            # Generate character_id from character name
            char_id = char_name_to_id[rel.character]

            # Generate relationship_id: {character_id}-{type}
            rel_id = f"{char_id}-{rel.type}"

            # Check for duplicates
            if rel_id in seen_ids:
                raise ValueError(
                    f"Duplicate relationship definition: character='{rel.character}', type='{rel.type}'. "
                    f"Each character can only have one relationship per type."
                )
            seen_ids.add(rel_id)

            # Create processed relationship with ID
            processed_rel = StoryCharacterRelationship(
                character=rel.character,
                type=rel.type,
                points_per_level=rel.points_per_level,
                max_levels=rel.max_levels,
                starting_points=rel.starting_points,
                visual=rel.visual,
                relationship_id=rel_id,
            )
            processed.append(processed_rel)

            logger.debug(f"  RELATIONSHIP: {rel.character} ({rel.type}) -> {rel_id}")

        logger.info(f"SUCCESS: Processed {len(processed)} relationship(s)")
        return processed

    def _validate_relationship_effects(
        self,
        instructions: List[StoryInstruction],
        relationships: List[StoryCharacterRelationship],
    ) -> None:
        """
        Validate that all relationshipEffects in choices reference valid relationships.

        Args:
            instructions: Story instructions
            relationships: Processed relationships

        Raises:
            ValueError: If a choice references an invalid relationship
        """
        if not relationships:
            # If no relationships defined, ensure no choices have relationshipEffects
            for instr in instructions:
                if instr.choices:
                    for choice in instr.choices:
                        if choice.relationship_effects:
                            raise ValueError(
                                f"Choice {choice.id} has relationshipEffects, but story has no relationships defined. "
                                f"Add relationships to the story's 'relationships' array."
                            )
            return

        # Build lookup of valid (character, type) pairs
        valid_relationships = {(rel.character, rel.type) for rel in relationships}

        for instr in instructions:
            if not instr.choices:
                continue

            for choice in instr.choices:
                if not choice.relationship_effects:
                    continue

                for effect in choice.relationship_effects:
                    if (effect.character, effect.type) not in valid_relationships:
                        raise ValueError(
                            f"Invalid relationship effect in choice {choice.id}: "
                            f"No relationship defined for character '{effect.character}' with type '{effect.type}'. "
                            f"Available relationships: {valid_relationships}. "
                            f"Add this relationship to the story's 'relationships' array."
                        )

        logger.debug("SUCCESS: All relationshipEffects validated successfully")

    def _parse_scenes(
        self,
        instructions: List[StoryInstruction],
        characters: List[StoryCharacter],
    ) -> List[Scene]:
        """
        Parse instructions into scenes.
        
        A scene is a group of instructions between 'setting' instructions.
        
        Args:
            instructions: Story instructions
            characters: Processed characters
            
        Returns:
            List of scenes
        """
        logger.debug("PARSING: Parsing scenes from instructions...")
        
        scenes = []
        current_scene = None
        scene_number = 0
        global_beat_number = 0  # Q1: Track global beat counter across all scenes
        
        for idx, instruction in enumerate(instructions):
            if instruction.type == "setting":
                # Save previous scene if exists
                if current_scene:
                    scenes.append(current_scene)
                
                # Start new scene
                scene_number += 1
                scene_id = f"scene-{scene_number}"
                
                current_scene = Scene(
                    scene_id=scene_id,
                    scene_number=scene_number,
                    title=instruction.title or f"Scene {scene_number}",
                    location=instruction.setting or "Unknown location",
                    instructions=[instruction],  # FIX: Include the setting instruction itself!
                    characters=[],
                    dialogue_beats=[],
                )
                
                logger.debug(f"  SCENE: Scene {scene_number}: {current_scene.title}")
            
            elif instruction.type == "end":
                # End of story
                if current_scene:
                    scenes.append(current_scene)
                break
            
            else:
                # Add instruction to current scene
                if current_scene:
                    current_scene.instructions.append(instruction)
                    
                    # Track characters in this scene
                    if instruction.character:
                        # Use character name (lowercase) for scene tracking
                        char_name_lower = instruction.character.lower()
                        if char_name_lower not in current_scene.characters:
                            current_scene.characters.append(char_name_lower)
                            logger.debug(f"      👤 Added character to scene: {instruction.character}")
                    
                    # Track dialogue beats
                    if instruction.type == "dialogue":
                        # Q1: Increment global beat counter
                        global_beat_number += 1
                        
                        # Scene-local beat number
                        scene_beat_number = len(current_scene.dialogue_beats) + 1
                        beat_id = f"{current_scene.scene_id}-beat-{scene_beat_number}"
                        
                        # Q2: Use manual topic if provided, otherwise extract from text
                        topic = instruction.topic if instruction.topic else self._extract_topic(instruction.text or "")
                        
                        # Q4: Get priority and dependencies
                        priority = instruction.priority or "required"  # Default to required
                        requires_beats = instruction.requires_beats or []  # Default to no dependencies
                        
                        dialogue_beat = {
                            "beat_id": beat_id,
                            "beat_number": scene_beat_number,  # Local to scene
                            "global_beat_number": global_beat_number,  # Q1: Global counter
                            "character": instruction.character,
                            "script_text": instruction.text,
                            "topic": topic,  # Q2: Manual or extracted
                            "priority": priority,  # Q4: required or optional
                            "requires_beats": requires_beats,  # Q4: Dependencies
                            "is_completed": False,
                            # Q9: Beat metadata enrichment
                            "emotion": instruction.emotion,
                            "animation": instruction.animation,
                            "camera_angle": instruction.camera_angle,
                            "timing_hint": instruction.timing_hint,
                            "sfx": instruction.sfx,
                            "music_cue": instruction.music_cue,
                            # Multiple choice support (Kon Unity integration)
                            "choices": instruction.choices,
                        }
                        current_scene.dialogue_beats.append(dialogue_beat)
                        
                        logger.debug(
                            f"    DIALOGUE: Beat {beat_id} (global #{global_beat_number}): "
                            f"{instruction.character} - {topic[:30]}... [{priority}]"
                            + (f" emotion={instruction.emotion}" if instruction.emotion else "")
                        )
                    
                    # Q5: Track narration beats (checkpoints)
                    elif instruction.type == "narration":
                        global_beat_number += 1
                        narration_number = len(current_scene.narration_beats) + 1
                        narration_id = f"{current_scene.scene_id}-narration-{narration_number}"
                        
                        priority = instruction.priority or "required"
                        requires_beats = instruction.requires_beats or []
                        
                        narration_beat = {
                            "beat_id": narration_id,
                            "beat_number": narration_number,
                            "global_beat_number": global_beat_number,
                            "text": instruction.text,
                            "priority": priority,
                            "requires_beats": requires_beats,
                            "is_completed": False,
                            # Q9: Metadata
                            "camera_angle": instruction.camera_angle,
                            "timing_hint": instruction.timing_hint,
                            "sfx": instruction.sfx,
                            "music_cue": instruction.music_cue,
                            # Multiple choice support (Kon Unity integration)
                            "choices": instruction.choices,
                        }
                        # Track characters mentioned in narration choices (for scene presence)
                        if instruction.choices:
                            for choice in instruction.choices:
                                if hasattr(choice, 'relationshipEffects') and choice.relationshipEffects:
                                    for effect in choice.relationshipEffects:
                                        if hasattr(effect, 'character') and effect.character:
                                            char_name_lower = effect.character.lower()
                                            if char_name_lower not in current_scene.characters:
                                                current_scene.characters.append(char_name_lower)
                                                logger.debug(f"      👤 Added character from choice effect: {effect.character}")
                        
                        logger.info(f"    DEBUG: Created narration_beat with choices={narration_beat.get('choices')}")
                        current_scene.narration_beats.append(narration_beat)
                        logger.info(f"    DEBUG: Scene now has {len(current_scene.narration_beats)} narration beats")
                        
                        logger.info(
                            f"    NARRATION: Narration {narration_id} (global #{global_beat_number}): "
                            f"{instruction.text[:30] if instruction.text else 'N/A'}... [{priority}] "
                            f"CHOICES={instruction.choices}"
                        )
                    
                    # Q5: Track action beats (checkpoints)
                    elif instruction.type == "action":
                        global_beat_number += 1
                        action_number = len(current_scene.action_beats) + 1
                        action_id = f"{current_scene.scene_id}-action-{action_number}"
                        
                        priority = instruction.priority or "required"
                        requires_beats = instruction.requires_beats or []
                        
                        action_beat = {
                            "beat_id": action_id,
                            "beat_number": action_number,
                            "global_beat_number": global_beat_number,
                            "character": instruction.character,
                            "action_text": instruction.action or instruction.text,
                            "priority": priority,
                            "requires_beats": requires_beats,
                            "is_completed": False,
                            # Q9: Metadata
                            "animation": instruction.animation,
                            "camera_angle": instruction.camera_angle,
                            "timing_hint": instruction.timing_hint,
                            "sfx": instruction.sfx,
                            # Multiple choice support (Kon Unity integration)
                            "choices": instruction.choices,
                        }
                        current_scene.action_beats.append(action_beat)
                        
                        logger.debug(
                            f"    ACTION: Action {action_id} (global #{global_beat_number}): "
                            f"{instruction.character or 'N/A'} - {(instruction.action or instruction.text or '')[:30]}... [{priority}]"
                        )
        
        # Add last scene if exists
        if current_scene and current_scene not in scenes:
            # If no characters specified, default to all NPCs
            if not current_scene.characters:
                npc_names = [char.name.lower() for char in characters if not char.is_main_character]
                current_scene.characters = npc_names
                logger.info(f"  INFO: Scene {current_scene.scene_number} had no explicit characters, defaulting to all NPCs: {npc_names}")
            
            scenes.append(current_scene)
        
        # Apply smart defaults to all scenes (in case some were saved earlier)
        for scene in scenes:
            if not scene.characters:
                npc_names = [char.name.lower() for char in characters if not char.is_main_character]
                scene.characters = npc_names
                logger.info(f"  INFO: Scene {scene.scene_number} had no explicit characters, defaulting to all NPCs: {npc_names}")
        
        # Count total beats by type (Q5)
        total_dialogue = sum(len(scene.dialogue_beats) for scene in scenes)
        total_narration = sum(len(scene.narration_beats) for scene in scenes)
        total_action = sum(len(scene.action_beats) for scene in scenes)
        
        logger.debug(
            f"SUCCESS: Parsed {len(scenes)} scenes with {global_beat_number} total checkpoints "
            f"(Dialogue: {total_dialogue}, Narration: {total_narration}, Actions: {total_action})"
        )
        return scenes

    def _extract_topic(self, text: str) -> str:
        """
        Extract topic from dialogue text (simple keyword extraction).
        
        Args:
            text: Dialogue text
            
        Returns:
            Topic summary
        """
        # Simple topic extraction - take first 50 chars
        # TODO: Could use NLP for better topic extraction
        topic = text[:50]
        if len(text) > 50:
            topic += "..."
        return topic

    def _validate_scenes(self, scenes: List[Scene]) -> None:
        """
        Validate scene structure.
        
        Args:
            scenes: Parsed scenes
            
        Raises:
            ValueError: Invalid scene structure
        """
        if not scenes:
            raise ValueError("Story must have at least one scene")
        
        for scene in scenes:
            if not scene.title:
                raise ValueError(f"Scene {scene.scene_number} missing title")
            
            if not scene.location:
                raise ValueError(f"Scene {scene.scene_number} missing location")
            
            if not scene.instructions:
                logger.warning(f"WARNING: Scene {scene.scene_number} has no instructions")
        
        logger.debug(f"SUCCESS: Scene validation passed ({len(scenes)} scenes)")
