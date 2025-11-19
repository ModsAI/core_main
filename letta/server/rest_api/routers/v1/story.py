"""
Story System API Router

Provides endpoints for:
- Story upload and retrieval
- Session management (start, resume, restart, delete)
- Dialogue generation (coming soon after Q6-Q10 answered)
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import JSONResponse

from letta.log import get_logger
from letta.schemas.story import (
    AdvanceStoryResponse,
    RelationshipStatusResponse,
    SessionCreate,
    SessionRestartResponse,
    SessionResume,
    SessionStartResponse,
    SessionStateResponse,
    StoryChoiceRequest,
    StoryChoiceResponse,
    StoryDetailResponse,
    StoryDialogueRequest,
    StoryDialogueResponse,
    StoryError,
    StoryListItem,
    StoryListResponse,
    StoryUpload,
    StoryUploadResponse,
)
from letta.server.db import db_registry
from letta.server.rest_api.utils import get_letta_server
from letta.server.server import SyncServer  # Import for type hints (use string annotation in function signatures)
from letta.services.session_manager import SessionManager
from letta.services.story_manager import StoryManager

logger = get_logger(__name__)

router = APIRouter(prefix="/story", tags=["story"])


# ============================================================
# Story Upload & Management
# ============================================================


@router.post("/upload", response_model=StoryUploadResponse)
async def upload_story(
    story: StoryUpload,
    overwrite: bool = False,
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Upload a new story or update an existing one.
    
    Converts TypeScript story JSON to internal format and stores in database.
    
    **Process:**
    1. Validates story structure
    2. Generates character IDs
    3. Parses instructions into scenes
    4. Extracts dialogue beats
    5. Stores in database
    6. If `overwrite=true`, deletes existing story first
    
    **Example Request:**
    ```json
    {
        "id": 1001,
        "title": "The Inexperienced Me",
        "description": "A coming-of-age story",
        "characters": [
            {
                "name": "Yuki",
                "sex": "male",
                "age": 17,
                "isMainCharacter": true
            },
            {
                "name": "Tatsuya",
                "sex": "male",
                "age": 17
            }
        ],
        "instructions": [
            {
                "type": "setting",
                "title": "Scene 1",
                "setting": "School courtyard"
            },
            {
                "type": "dialogue",
                "character": "Tatsuya",
                "text": "Hey, what's up?"
            },
            {
                "type": "end"
            }
        ]
    }
    ```
    
    **Returns:**
    - Story ID
    - Scene count
    - Character count
    - Next steps
    
    **Errors:**
    - 400: Invalid story structure
    - 409: Story ID already exists
    - 500: Database error
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=actor_id)
    logger.info(f"📤 Story upload request: {story.title} (user: {actor.id}, overwrite={overwrite})")
    
    try:
        story_manager = StoryManager()
        
        # If overwrite=true, delete existing story first
        sessions_deleted = 0
        if overwrite:
            story_id = f"story-{story.id}"
            existing = await story_manager.get_story(story_id, actor)
            if existing:
                logger.info(f"  🔄 Overwrite mode: Deleting existing story {story_id}")
                
                # Count sessions before deleting
                from sqlalchemy import select, func
                from letta.orm.story import StorySession as StorySessionORM
                async with db_registry.async_session() as count_session:
                    count_stmt = select(func.count(StorySessionORM.id)).where(
                        StorySessionORM.story_id == story_id,
                        StorySessionORM.organization_id == actor.organization_id,
                    )
                    count_result = await count_session.execute(count_stmt)
                    sessions_deleted = count_result.scalar() or 0
                
                if sessions_deleted > 0:
                    logger.warning(
                        f"  ⚠️ IMPORTANT: {sessions_deleted} existing session(s) will be deleted "
                        f"when story '{story_id}' is overwritten!"
                    )
                
                # Delete story (CASCADE will delete sessions automatically)
                deleted = await story_manager.delete_story(story_id, actor)
                
                if deleted:
                    logger.info(f"  ✅ Story deleted: {story_id} ({sessions_deleted} sessions invalidated)")
                else:
                    logger.error(f"  ❌ Failed to delete story: {story_id}")
            else:
                logger.info(f"  ℹ️ No existing story found for {story_id}, proceeding with fresh upload")
        
        response = await story_manager.upload_story(story, actor)
        
        # Add session invalidation warning to response if sessions were deleted
        if sessions_deleted > 0:
            response.instructions.insert(0, 
                f"⚠️ IMPORTANT: {sessions_deleted} existing session(s) were invalidated during overwrite"
            )
            response.instructions.insert(1,
                "Any old session IDs are now invalid - you MUST start a NEW session"
            )
            response.instructions.insert(2,
                "Do NOT reuse old session IDs after overwrite - they have been deleted"
            )
        
        logger.info(f"✅ Story uploaded: {response.story_id} (overwrite={overwrite}, sessions_deleted={sessions_deleted})")
        return response
    
    except ValueError as e:
        logger.error(f"❌ Story upload validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "INVALID_STORY",
                "message": str(e),
                "suggestions": [
                    "Check that story has valid structure",
                    "Ensure all required fields are present",
                    "Verify characters have unique names",
                    "Confirm instructions include at least one scene",
                ],
            },
        )
    
    except Exception as e:
        error_msg = str(e)
        
        # Check for duplicate story
        if "already exists" in error_msg.lower():
            logger.error(f"❌ Duplicate story ID: {story.id}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": "STORY_EXISTS",
                    "message": f"Story with ID {story.id} already exists",
                    "suggestions": [
                        "Use a different story ID",
                        "Delete the existing story first",
                        "Update the existing story instead",
                    ],
                },
            )
        
        # Generic error
        logger.error(f"❌ Story upload error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "UPLOAD_FAILED",
                "message": f"Failed to upload story: {error_msg}",
                "suggestions": [
                    "Check server logs for details",
                    "Verify database connection",
                    "Try again in a moment",
                ],
            },
        )


@router.get("/list", response_model=StoryListResponse)
async def list_stories(
    page: int = 1,
    page_size: int = 50,
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    List all available stories with metadata.
    
    Returns a paginated list of stories for story selection menus.
    Includes scene count, character count, and estimated duration.
    
    Query Parameters:
    - page: Page number (default: 1)
    - page_size: Items per page (default: 50, max: 100)
    
    Returns:
    - List of stories with metadata
    - Total count for pagination
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=actor_id)
    
    try:
        logger.info(f"📋 Listing stories (user={actor_id}, page={page}, page_size={page_size})")
        
        # Validate pagination parameters
        if page < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Page number must be >= 1"
            )
        if page_size < 1 or page_size > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Page size must be between 1 and 100"
            )
        
        # Get story manager
        story_manager = StoryManager()
        
        # List stories
        stories, total = await story_manager.list_stories(
            actor=actor,
            page=page,
            page_size=page_size,
        )
        
        # Convert to Pydantic models
        story_items = [StoryListItem(**story) for story in stories]
        
        logger.info(f"✅ Returning {len(story_items)} stories (total: {total})")
        
        return StoryListResponse(
            stories=story_items,
            total=total,
            page=page,
            page_size=page_size,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to list stories: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list stories: {str(e)}",
        )


# ============================================================
# Session Management
# ============================================================


@router.post("/sessions/start", response_model=SessionStartResponse)
async def start_session(
    session_create: SessionCreate,
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Start a new story session.
    
    **Important:**
    - Only ONE active session per story per user
    - Starting a new session will DELETE any existing session for that story
    - This creates Letta agents for all characters (may take 10-30 seconds)
    
    **Process:**
    1. Retrieves story from database
    2. Deletes any existing session for this story
    3. Creates Letta agents for all characters
    4. Initializes session state
    5. Returns first scene
    
    **Example Request:**
    ```json
    {
        "story_id": "story-1001"
    }
    ```
    
    **Returns:**
    - Session ID
    - First scene
    - Player character name
    - Instructions for next steps
    
    **Errors:**
    - 404: Story not found
    - 500: Agent creation or database error
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=actor_id)
    logger.info(f"🎬 Session start request: {session_create.story_id} (user: {actor.id})")
    
    try:
        session_manager = SessionManager()
        response = await session_manager.start_session(session_create, actor)
        
        logger.info(f"✅ Session started: {response.session_id}")
        return response
    
    except ValueError as e:
        logger.error(f"❌ Session start validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "STORY_NOT_FOUND",
                "message": str(e),
                "suggestions": [
                    f"Verify story ID '{session_create.story_id}' exists",
                    "Check if you have permission to access this story",
                    "Upload the story first with POST /api/v1/story/upload",
                ],
            },
        )
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Session start error: {e}", exc_info=True)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "SESSION_START_FAILED",
                "message": f"Failed to start session: {error_msg}",
                "suggestions": [
                    "Check server logs for agent creation errors",
                    "Verify database connection",
                    "Ensure Gemini API key is configured",
                    "Try again in a moment",
                ],
            },
        )


@router.post("/sessions/resume", response_model=SessionResume)
async def resume_session(
    session_resume: SessionCreate,  # Changed: now takes story_id in body
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Resume an existing session.
    
    Finds and resumes the active session for a given story.
    
    **Example Request:**
    ```json
    {
        "story_id": "story-1005"
    }
    ```
    
    **Returns:**
    - Session details
    - Current scene
    - Current progress
    - Recent chat history
    
    **Errors:**
    - 404: No active session found for this story
    - 500: Database error
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=actor_id)
    logger.info(f"▶️ Session resume request: {session_resume.story_id} (user: {actor.id})")
    
    try:
        session_manager = SessionManager()
        response = await session_manager.resume_session(session_resume.story_id, actor, server)
        
        logger.info(f"✅ Session resumed for story: {session_resume.story_id}")
        return response
    
    except ValueError as e:
        logger.error(f"❌ Session resume validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "SESSION_NOT_FOUND",
                "message": str(e),
                "suggestions": [
                    f"No active session found for story '{session_resume.story_id}'",
                    "Start a new session with POST /api/v1/story/sessions/start",
                    "Session may have been deleted or expired",
                ],
            },
        )
    
    except Exception as e:
        logger.error(f"❌ Session resume error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "SESSION_RESUME_FAILED",
                "message": f"Failed to resume session: {str(e)}",
                "suggestions": [
                    "Check server logs for details",
                    "Verify database connection",
                    "Try again in a moment",
                ],
            },
        )


@router.post("/sessions/{session_id}/restart", response_model=SessionRestartResponse)
async def restart_session(
    session_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Restart a session from the beginning.
    
    **Warning: This is DESTRUCTIVE!**
    - Deletes all character agents
    - Wipes all progress
    - Creates new agents
    - Starts from scene 1
    
    **Example:**
    ```
    POST /api/v1/story/sessions/session-abc123/restart
    ```
    
    **Returns:**
    - New session ID
    - Success message
    
    **Errors:**
    - 404: Session not found
    - 500: Deletion or creation error
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=actor_id)
    logger.info(f"🔄 Session restart request: {session_id} (user: {actor.id})")
    
    try:
        session_manager = SessionManager()
        response = await session_manager.restart_session(session_id, actor)
        
        logger.info(f"✅ Session restarted: {response.session_id}")
        return response
    
    except ValueError as e:
        logger.error(f"❌ Session restart validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "SESSION_NOT_FOUND",
                "message": str(e),
                "suggestions": [
                    f"Verify session ID '{session_id}' exists",
                    "Session may have already been deleted",
                ],
            },
        )
    
    except Exception as e:
        logger.error(f"❌ Session restart error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "SESSION_RESTART_FAILED",
                "message": f"Failed to restart session: {str(e)}",
                "suggestions": [
                    "Check server logs for details",
                    "Verify database connection",
                    "Try starting a new session instead",
                ],
            },
        )


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Delete a session permanently.
    
    **Warning: This is PERMANENT!**
    - Deletes all character agents
    - Removes session from database
    - Cannot be undone
    
    **Example:**
    ```
    DELETE /api/v1/story/sessions/session-abc123
    ```
    
    **Returns:**
    - Success message
    
    **Errors:**
    - 404: Session not found
    - 500: Deletion error
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=actor_id)
    logger.info(f"🗑️ Session delete request: {session_id} (user: {actor.id})")
    
    try:
        session_manager = SessionManager()
        deleted = await session_manager.delete_session(session_id, actor)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "SESSION_NOT_FOUND",
                    "message": f"Session '{session_id}' not found",
                },
            )
        
        logger.info(f"✅ Session deleted: {session_id}")
        return {"success": True, "message": f"Session {session_id} deleted successfully"}
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"❌ Session delete error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "SESSION_DELETE_FAILED",
                "message": f"Failed to delete session: {str(e)}",
                "suggestions": [
                    "Check server logs for details",
                    "Verify database connection",
                    "Try again in a moment",
                ],
            },
        )


# ============================================================
# Dialogue Generation
# ============================================================


@router.post("/sessions/{session_id}/dialogue", response_model=StoryDialogueResponse)
async def generate_dialogue(
    session_id: str,
    request: StoryDialogueRequest,
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Generate character dialogue with script guidance.
    
    **How It Works:**
    1. Player sends message to character
    2. System gets next dialogue beat from script
    3. Agent responds naturally while covering script topic
    4. Beat marked as completed
    5. Scene auto-advances when all beats done
    
    **Dialogue Behavior (Per Requirements):**
    - **Script Guidance (Q6):** Agent uses script as CONTEXT, responds naturally, guides conversation forward
    - **Off-Topic (Q7):** Agent answers briefly, then redirects to script
    - **Scene Progression (Q8):** Automatic when all dialogue beats completed
    - **Skipping (Q9):** Not allowed - must complete all beats
    - **Error Fallback (Q10):** Uses scripted text if agent fails, retries automatically
    
    **Example Request:**
    ```json
    {
        "player_message": "Hey Tatsuya, what's up?",
        "target_character": "tatsuya"
    }
    ```
    
    **Example Response:**
    ```json
    {
        "character_id": "tatsuya",
        "character_name": "Tatsuya",
        "dialogue_text": "Hey man! Not much. You know, you really can't keep living in that phone forever. There's a whole world out here.",
        "emotion": "friendly",
        "animation_suggestion": "gesture_casual",
        "dialogue_beats_completed": ["scene-1-beat-1"],
        "scene_progress": 0.5,
        "scene_complete": false,
        "session_updated": true,
        "next_scene_number": null
    }
    ```
    
    **Returns:**
    - Character response (natural, script-guided)
    - Emotion (for Unity facial expression)
    - Animation suggestion (for Unity animator)
    - Dialogue beats completed (story progress)
    - Scene progress (0.0 to 1.0)
    - Scene complete flag
    - Next scene number (if auto-advanced)
    
    **Errors:**
    - 404: Session not found
    - 404: Character not found
    - 500: Dialogue generation error
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=actor_id)
    logger.info(f"💬 Dialogue request: session={session_id}, character={request.target_character}, user={actor.id}")
    
    try:
        from letta.services.dialogue_manager import DialogueManager
        
        logger.debug(f"  🔧 Creating DialogueManager...")
        dialogue_manager = DialogueManager(server=server)
        
        logger.debug(f"  🚀 Calling generate_dialogue...")
        response = await dialogue_manager.generate_dialogue(session_id, request, actor)
        
        logger.info(f"✅ Dialogue generated: {len(response.dialogue_text)} chars, emotion={response.emotion}")
        return response
    
    except ValueError as e:
        logger.error(f"❌ Dialogue validation error: {e}")
        logger.error(f"   Session: {session_id}, Character: {request.target_character}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "DIALOGUE_ERROR",
                "message": str(e),
                "suggestions": [
                    "Verify session ID exists",
                    "Check character ID is valid",
                    "Ensure session is active",
                ],
            },
        )
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Dialogue generation error: {e}", exc_info=True)
        logger.error(f"   Session: {session_id}, Character: {request.target_character}, Error type: {type(e).__name__}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "DIALOGUE_GENERATION_FAILED",
                "message": f"Failed to generate dialogue: {error_msg}",
                "suggestions": [
                    "Check server logs for agent errors",
                    "Verify Gemini API key is configured",
                    "Ensure agent exists for character",
                    "Try again in a moment",
                ],
            },
        )


@router.post("/sessions/{session_id}/select-choice", response_model=StoryChoiceResponse)
async def select_choice(
    session_id: str,
    request: StoryChoiceRequest,
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Handle player choice selection (for multiple choice questions).

    **Purpose:**
    This endpoint handles when the player makes a choice selection (tap/click)
    instead of typing dialogue. This is for:
    - Multiple choice questions in narration
    - Dialogue choice branches
    - Quick time events
    - Any non-dialogue player input

    **How It Works:**
    1. Player selects a choice by ID
    2. Choice is recorded in session state
    3. Current instruction (with choices) is marked complete
    4. Story advances to next instruction
    5. Session state is updated

    **Use Cases:**
    - Multiple choice questions: "What path will you take?"
    - Dialogue branches: "How do you respond?"
    - Quick decisions: "Save teammate or pursue enemy?"

    **Example Request:**
    ```json
    {
        "choice_id": 2,
        "choice_text": "Investigate the artifact"
    }
    ```

    **Example Response:**
    ```json
    {
        "success": true,
        "choice_id": 2,
        "message": "Choice recorded and story advanced",
        "session_id": "session-abc-123",
        "timestamp": "2025-10-30T16:45:00Z"
    }
    ```

    **Returns:**
    - Success status
    - Choice ID that was selected
    - Status message
    - Session ID
    - Timestamp

    **After this call:**
    - Call GET /sessions/{session_id}/state to get next instruction
    - The next instruction will be whatever follows the choice in the script

    **Errors:**
    - 404: Session not found
    - 400: Current instruction doesn't have choices
    - 500: Database error
    """
    from datetime import datetime

    actor = await server.user_manager.get_actor_or_default_async(actor_id=actor_id)
    logger.info(f"🎯 Choice selection: session={session_id}, choice_id={request.choice_id}")

    try:
        from sqlalchemy import select, update

        from letta.orm.story import StorySession as StorySessionORM
        from letta.services.session_manager import SessionManager
        from letta.services.story_manager import StoryManager

        session_manager = SessionManager()
        story_manager = StoryManager()

        # Get session and story
        session = await session_manager._get_session_by_id(session_id, actor)
        if not session:
            raise ValueError(f"Session '{session_id}' not found")

        story = await story_manager.get_story(session.story_id, actor)
        if not story:
            raise ValueError(f"Story '{session.story_id}' not found")

        # Store original version for optimistic locking
        # FIX: Handle NULL version (same as advance-story fix)
        original_version = session.version if session.version is not None else 1
        
        # If version was NULL, initialize it to 1 immediately
        if session.version is None:
            logger.warning(f"  ⚠️ Session {session_id} has NULL version, initializing to 1")
            async with db_registry.async_session() as init_session:
                async with init_session.begin():
                    stmt = update(StorySessionORM).where(
                        StorySessionORM.session_id == session_id,
                        StorySessionORM.version.is_(None)  # Use IS NULL check
                    ).values(version=1)
                    await init_session.execute(stmt)
            # Re-fetch session with updated version
            session = await session_manager._get_session_by_id(session_id, actor)
            original_version = 1

        # Get the current beat (the one with choices) so we can mark it as completed
        current_beat = session_manager._get_next_instruction(story, session.state)
        
        # Get selected choice object for metadata
        selected_choice = None
        if current_beat and current_beat.get("choices"):
            for choice in current_beat.get("choices", []):
                if choice.get("id") == request.choice_id:
                    selected_choice = choice
                    break
        
        # Debug logging
        if not selected_choice:
            logger.warning(f"  ⚠️ Could not find choice {request.choice_id} in current beat")
            logger.warning(f"     Current beat type: {current_beat.get('type') if current_beat else 'None'}")
            logger.warning(f"     Current beat has choices: {bool(current_beat.get('choices')) if current_beat else False}")
            if current_beat and current_beat.get("choices"):
                choice_ids = [c.get("id") for c in current_beat.get("choices", [])]
                logger.warning(f"     Available choice IDs: {choice_ids}")
        
        # Record choice in session state with enhanced metadata
        choice_record = {
            "choice_id": request.choice_id,
            "choice_text": request.choice_text,
            "scene_number": session.state.current_scene_number,
            "instruction_type": current_beat.get("type", "") if current_beat else "",
            "question_context": current_beat.get("text", "")[:100] if current_beat else "",  # Truncate to 100 chars
            "relationship_effects": selected_choice.get("relationship_effects", []) if selected_choice else [],
            "timestamp": datetime.now().isoformat(),
        }
        session.state.player_choices.append(choice_record)

        # NEW: Apply relationship effects from the choice
        session_manager.apply_relationship_effects(
            session_state=session.state,
            story=story,
            choice_id=request.choice_id,
            current_instruction=current_beat,
        )

        # FIX: Mark the current beat as completed (this was missing!)
        if current_beat and current_beat.get("beat_id"):
            beat_id = current_beat["beat_id"]
            beat_type = current_beat.get("type")
            
            if beat_type == "dialogue":
                if beat_id not in session.state.completed_dialogue_beats:
                    session.state.completed_dialogue_beats.append(beat_id)
                    logger.info(f"  ✅ Completed dialogue beat: {beat_id}")
            elif beat_type == "narration":
                if beat_id not in session.state.completed_narration_beats:
                    session.state.completed_narration_beats.append(beat_id)
                    logger.info(f"  ✅ Completed narration beat: {beat_id}")
            elif beat_type == "action":
                if beat_id not in session.state.completed_action_beats:
                    session.state.completed_action_beats.append(beat_id)
                    logger.info(f"  ✅ Completed action beat: {beat_id}")

        # Advance to next instruction
        current_scene = story.scenes[session.state.current_scene_number - 1]
        session.state.current_instruction_index += 1

        logger.info(f"  ✓ Choice recorded: {request.choice_id}, advanced to instruction {session.state.current_instruction_index}")

        # Save session state with optimistic locking (FIXED - use proper method)
        update_result = await session_manager.update_session_state_with_version(
            session_id=session_id,
            state=session.state,
            expected_version=original_version,
            actor=actor,
        )
        
        if not update_result["success"]:
            # Version mismatch - state was modified concurrently
            logger.error(f"  ❌ Concurrent modification detected during choice selection")
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "CONCURRENT_MODIFICATION",
                    "message": "State was modified by another request. Please try again.",
                    "suggestions": ["Try selecting the choice again", "Refresh session state"],
                },
            )

        # NEW: Update AI agents' relationship memory blocks to keep them synchronized
        await session_manager.update_relationships_memory(
            session_id=session_id,
            story=story,
            session_state=session.state,
            character_agents=session.character_agents,
            actor=actor,
        )

        logger.info(f"✅ Choice selection complete: {request.choice_id}")

        # Include updated relationship status in response (NEW)
        return StoryChoiceResponse(
            success=True,
            choice_id=request.choice_id,
            message="Choice recorded and story advanced",
            session_id=session_id,
            timestamp=datetime.now(),
            relationship_points=session.state.relationship_points,  # Include updated points
            relationship_levels=session.state.relationship_levels,  # Include updated levels
        )

    except ValueError as e:
        logger.error(f"❌ Choice selection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "CHOICE_ERROR", "message": str(e), "suggestions": ["Verify session exists", "Check story is loaded"]},
        )

    except Exception as e:
        logger.error(f"❌ Choice selection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "CHOICE_PROCESSING_FAILED",
                "message": f"Failed to process choice: {str(e)}",
                "suggestions": ["Check server logs", "Try again"],
            },
        )


@router.post("/sessions/{session_id}/advance-story", response_model=AdvanceStoryResponse)
async def advance_story(
    session_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Advance story to next instruction (player "tap to continue").

    **Purpose:**
    Allows player to continue/advance the story without typing custom messages.
    Supports BOTH tap-through dialogue and non-interactive beats.

    **Use Cases:**
    - **Dialogue beats** → Player taps to use scripted text (tap-through mode)
    - **Narration beats** → Player taps "Continue"
    - **Action beats** → Player taps to acknowledge
    - **Setting descriptions** → Player taps to proceed
    - Any instruction that doesn't require player input

    **How It Works:**
    1. Get current session state
    2. Get current instruction from `next_instruction`
    3. Mark current beat as completed
    4. Return updated state with next instruction

    **Difference from other endpoints:**
    - `/select-choice`: For when player chooses from options
    - `/dialogue`: For custom player messages (AI-generated responses)
    - `/advance-story`: For tap-through (uses scripted text, no AI)
    - `/skip-beat`: For QA/admin to force-skip (requires beat_id)
    
    **Dialogue Modes (NEW):**
    - **Tap-through** (`/advance-story`): Fast, uses scripted text, no typing
    - **Custom** (`/dialogue`): Player types message, AI generates response

    **Example Request:**
    ```
    POST /v1/story/sessions/session-abc-123/advance-story
    ```

    **Example Response:**
    ```json
    {
        "success": true,
        "advanced_from": "scene-1-narration-1",
        "beat_type": "narration",
        "message": "Advanced from narration beat",
        "next_instruction": {
            "type": "dialogue",
            "character": "Hero",
            "text": "...",
            "choices": null
        },
        "session_id": "session-abc-123"
    }
    ```

    **Returns:**
    - Success status
    - Beat ID that was completed
    - Type of beat (narration/action/setting)
    - Next instruction to display
    - Session ID

    **Errors:**
    - 404: Session not found
    - 400: Beat has choices (use /select-choice)
    - 400: Story is complete
    - 409: Concurrent modification (auto-retried up to 3 times)
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=actor_id)
    logger.info(f"📍 Advance Story - Session: {session_id}, User: {actor.id}")

    # Retry logic for concurrent modifications (optimistic locking)
    MAX_RETRIES = 3
    
    for attempt in range(MAX_RETRIES):
        try:
            session_manager = SessionManager()
            story_manager = StoryManager()
            
            if attempt > 0:
                logger.info(f"  ♻️ Retry attempt {attempt + 1}/{MAX_RETRIES}")

            # Get session
            session = await session_manager._get_session_by_id(session_id, actor)
            if not session:
                logger.error(f"  ✗ Session not found: {session_id}")
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": "SESSION_NOT_FOUND",
                        "message": f"Session {session_id} not found",
                        "suggestions": ["Check session ID", "Ensure session was created"],
                    },
                )
            
            # Store original version for optimistic locking
            # FIX: Handle NULL version (existing sessions before migration)
            original_version = session.version if session.version is not None else 1
            
            # If version was NULL, initialize it to 1 immediately
            if session.version is None:
                logger.warning(f"  ⚠️ Session {session_id} has NULL version, initializing to 1")
                from sqlalchemy import update
                from letta.orm.story import StorySession as StorySessionORM
                async with db_registry.async_session() as init_session:
                    async with init_session.begin():
                        stmt = update(StorySessionORM).where(
                            StorySessionORM.session_id == session_id,
                            StorySessionORM.version.is_(None)  # Use IS NULL check
                        ).values(version=1)
                        await init_session.execute(stmt)
                # Re-fetch session with updated version
                session = await session_manager._get_session_by_id(session_id, actor)
                original_version = 1

            # Get story
            story = await story_manager.get_story(session.story_id, actor)
            if not story:
                logger.error(f"  ✗ Story not found: {session.story_id}")
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": "STORY_NOT_FOUND",
                        "message": f"Story {session.story_id} not found",
                        "suggestions": ["Check story ID", "Ensure story exists"],
                    },
                )

            # Get current instruction
            current_instruction = session_manager._get_next_instruction(story, session.state)

            if not current_instruction:
                logger.warning(f"  ⚠️ No current instruction to advance from")
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "NO_INSTRUCTION",
                        "message": "No current instruction to advance from. Story may be complete.",
                        "suggestions": ["Check if story has ended", "Use /state to verify current position"],
                    },
                )

            # Get beat details
            beat_type = current_instruction.get("type")
            beat_id = current_instruction.get("beat_id")

            logger.debug(f"  → Current instruction: type={beat_type}, beat_id={beat_id}")

            # NEW: Allow dialogue beats to be "tapped through" using scripted text
            # This supports Kon's requirement for BOTH tap-through and custom dialogue
            if beat_type == "dialogue":
                logger.info(f"  💬 Tap-through dialogue beat (using scripted text)")
                # Mark dialogue beat as completed
                if beat_id and beat_id not in session.state.completed_dialogue_beats:
                    session.state.completed_dialogue_beats.append(beat_id)
                    logger.info(f"  ✅ Marked dialogue beat as completed: {beat_id}")
                # Will advance instruction index below (same as other beats)

            # Check if this instruction has choices
            if current_instruction.get("choices"):
                logger.warning(f"  ⚠️ Instruction has choices - should use /select-choice")
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "HAS_CHOICES",
                        "message": "This instruction has choices. Use /select-choice endpoint instead.",
                        "choices": current_instruction.get("choices"),
                        "suggestions": ["Use POST /select-choice with choice_id", "Display choice buttons to player"],
                    },
                )

            # Check if story is complete
            if beat_type == "end":
                logger.info(f"  ✓ Story is complete")
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "STORY_COMPLETE",
                        "message": "Story has ended. No more instructions to advance to.",
                        "suggestions": ["Story is complete", "Can restart session with /restart"],
                    },
                )
            
            # Check if this is a scene transition (all beats complete in current scene)
            if beat_type == "setting" and current_instruction.get("is_completed"):
                # Scene is complete - advance to next scene automatically
                current_scene = session.state.current_scene_number
                if current_scene < len(story.scenes):
                    logger.info(f"  🎬 Scene {current_scene} complete - auto-advancing to Scene {current_scene + 1}")
                    session.state.current_scene_number = current_scene + 1
                    session.state.completed_dialogue_beats = []
                    session.state.completed_narration_beats = []
                    session.state.completed_action_beats = []
                    
                    # Save state with new scene
                    state_dict = session.state.model_dump(mode="json") if hasattr(session.state, "model_dump") else session.state.dict()
                    async with db_registry.async_session() as db_session:
                        async with db_session.begin():
                            from sqlalchemy import update
                            from letta.orm.story import StorySession as StorySessionORM
                            stmt = update(StorySessionORM).where(StorySessionORM.session_id == session_id).values(state=state_dict)
                            await db_session.execute(stmt)
                    
                    # Get next instruction from new scene
                    next_instruction = session_manager._get_next_instruction(story, session.state)
                    logger.info(f"  ✅ Advanced to Scene {current_scene + 1}")
                    
                    # Update NPC memory blocks with new scene context
                    try:
                        update_result = await session_manager.update_scene_memory_blocks(
                            session_id=session_id,
                            new_scene_number=current_scene + 1,
                            actor=actor,
                        )
                        logger.info(
                            f"  🧠 Updated {update_result['updated_agents']}/{update_result['total_agents']} "
                            f"agent memories for Scene {current_scene + 1}"
                        )
                    except Exception as e:
                        # Don't fail the entire scene transition if memory update fails
                        logger.error(f"  ⚠️ Failed to update scene memories: {e}", exc_info=True)
                    
                    return AdvanceStoryResponse(
                        success=True,
                        advanced_from=f"scene-{current_scene}",
                        beat_type="scene_transition",
                        message=f"Advanced to Scene {current_scene + 1}",
                        next_instruction=next_instruction,
                        session_id=session_id,
                    )
                else:
                    # ✅ FIX: Last scene complete - transition to "end" state
                    # This is Kon's issue: Core must send type: "end" instruction
                    logger.info(f"  🏁 Last scene complete - transitioning to story end")
                    
                    # Increment scene number to trigger "story complete" state
                    session.state.current_scene_number = current_scene + 1
                    
                    # Save state with story complete
                    state_dict = session.state.model_dump(mode="json") if hasattr(session.state, "model_dump") else session.state.dict()
                    async with db_registry.async_session() as db_session:
                        async with db_session.begin():
                            from sqlalchemy import update
                            from letta.orm.story import StorySession as StorySessionORM
                            stmt = update(StorySessionORM).where(StorySessionORM.session_id == session_id).values(state=state_dict)
                            await db_session.execute(stmt)
                    
                    # Get "end" instruction (scene_number > len(scenes) triggers this)
                    next_instruction = session_manager._get_next_instruction(story, session.state)
                    logger.info(f"  ✅ Story complete - returning type: 'end'")
                    
                    return AdvanceStoryResponse(
                        success=True,
                        advanced_from=f"scene-{current_scene}",
                        beat_type="story_complete",
                        message="Story has ended",
                        next_instruction=next_instruction,
                        session_id=session_id,
                    )

            # Mark beat as completed based on type
            if beat_id:
                if beat_type == "narration":
                    if beat_id not in session.state.completed_narration_beats:
                        session.state.completed_narration_beats.append(beat_id)
                        logger.info(f"  ✅ Marked narration beat as completed: {beat_id}")
                    else:
                        logger.debug(f"  ℹ️ Narration beat already completed: {beat_id}")
                elif beat_type == "action":
                    if beat_id not in session.state.completed_action_beats:
                        session.state.completed_action_beats.append(beat_id)
                        logger.info(f"  ✅ Marked action beat as completed: {beat_id}")
                    else:
                        logger.debug(f"  ℹ️ Action beat already completed: {beat_id}")
                else:
                    logger.debug(f"  ℹ️ Beat type {beat_type} doesn't need completion tracking")

            # FIX: Increment instruction index (same as /select-choice does)
            session.state.current_instruction_index += 1
            logger.debug(f"  → Instruction index advanced to {session.state.current_instruction_index}")

            # Save session state with optimistic locking
            update_result = await session_manager.update_session_state_with_version(
                session_id=session_id,
                state=session.state,
                expected_version=original_version,
                actor=actor,
            )
            
            if not update_result["success"]:
                # Version mismatch - state was modified concurrently
                if attempt < MAX_RETRIES - 1:
                    # Retry with exponential backoff
                    wait_time = 0.1 * (2 ** attempt)  # 0.1s, 0.2s, 0.4s
                    logger.warning(f"  ⚠️ Concurrent modification detected - retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue  # Retry the loop
                else:
                    # Max retries exceeded
                    logger.error(f"  ❌ Max retries exceeded - concurrent modifications ongoing")
                    return JSONResponse(
                        status_code=409,
                        content={
                            "error": "CONCURRENT_MODIFICATION",
                            "message": "State was modified by another request. Max retries exceeded.",
                            "suggestions": ["Try again", "Contact support if persists"],
                        },
                    )
            
            # Get next instruction
            next_instruction = session_manager._get_next_instruction(story, session.state)
            logger.debug(f"  → Next instruction: {next_instruction.get('type') if next_instruction else 'None'}")

            # Build response
            response = AdvanceStoryResponse(
                success=True,
                advanced_from=beat_id,
                beat_type=beat_type,
                message=f"Advanced from {beat_type} beat",
                next_instruction=next_instruction,
                session_id=session_id,
            )

            logger.info(f"  ✅ Story advanced successfully from {beat_type}")
            return response
            
        except ValueError as e:
            logger.error(f"❌ Advance Story FAILED (ValueError): {e}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "ADVANCE_FAILED",
                    "message": str(e),
                    "suggestions": ["Check session ID", "Verify story exists", "Check current instruction"],
                },
            )

        except Exception as e:
            logger.error(f"❌ Advance Story FAILED (Exception): {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "ADVANCE_FAILED",
                    "message": f"Failed to advance story: {str(e)}",
                    "suggestions": ["Check server logs", "Try again", "Contact support if issue persists"],
                },
            )
    
    # This should never be reached due to returns in loop, but just in case
    return JSONResponse(
        status_code=500,
        content={
            "error": "UNEXPECTED_ERROR",
            "message": "Unexpected error in retry loop",
        },
    )


@router.post("/sessions/{session_id}/advance-scene")
async def advance_scene_manually(
    session_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Manually advance to the next scene (Q6: Manual Override).

    **Purpose:**
    This endpoint allows Unity to manually force scene transition when:
    - Auto-advance fails or gets stuck
    - Player explicitly clicks "Next Scene" button
    - Scene is logically complete but system doesn't detect it

    **How It Works:**
    1. Gets current session state
    2. Increments scene number
    3. Resets beat progress for new scene
    4. Updates session in database

    **Use Case:**
    - Auto-advance didn't trigger (beat detection failed)
    - Player wants to skip to next scene
    - QA testing / debugging

    **Example Request:**
    ```
    POST /v1/story/sessions/session-abc-123/advance-scene
    ```

    **Example Response:**
    ```json
    {
        "success": true,
        "previous_scene": 1,
        "new_scene": 2,
        "message": "Advanced to Scene 2: The System's Whispers"
    }
    ```

    **Returns:**
    - Success status
    - Previous and new scene numbers
    - New scene title

    **Errors:**
    - 404: Session not found
    - 400: Already at last scene
    """
    from letta.services.session_manager import SessionManager

    actor = await server.user_manager.get_actor_or_default_async(actor_id=actor_id)
    logger.info(f"📍 Q6 Manual Scene Advance - Session: {session_id}, User: {actor.id}")

    try:
        session_manager = SessionManager()
        logger.debug(f"  ✓ SessionManager initialized")

        # Get current session
        session = await session_manager._get_session_by_id(session_id, actor)
        if not session:
            logger.warning(f"  ✗ Session not found: {session_id}")
            raise ValueError(f"Session '{session_id}' not found")
        logger.debug(f"  ✓ Session retrieved - Story: {session.story_id}, Current Scene: {session.state.current_scene_number}")

        # Get story
        from letta.services.story_manager import StoryManager

        story_manager = StoryManager()
        story = await story_manager.get_story(session.story_id, actor)

        if not story:
            logger.warning(f"  ✗ Story not found: {session.story_id}")
            raise ValueError(f"Story '{session.story_id}' not found")
        logger.debug(f"  ✓ Story retrieved - Title: '{story.title}', Total Scenes: {len(story.scenes)}")

        # Check if we can advance
        current_scene = session.state.current_scene_number
        logger.debug(f"  → Checking advancement: Current={current_scene}, Total={len(story.scenes)}")

        if current_scene >= len(story.scenes):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "SCENE_ADVANCE_ERROR",
                    "message": f"Already at last scene ({current_scene} of {len(story.scenes)})",
                    "suggestions": ["Story is complete", "Cannot advance further"],
                },
            )

        # Advance to next scene
        previous_scene = current_scene
        new_scene_num = current_scene + 1
        new_scene = story.scenes[new_scene_num - 1]
        logger.info(f"  ⏩ Advancing from Scene {previous_scene} to Scene {new_scene_num}: '{new_scene.title}'")

        # Update session state
        logger.debug(f"  → Updating session state...")
        session.state.current_scene_number = new_scene_num
        session.state.current_instruction_index = 0
        completed_beats_before = len(session.state.completed_dialogue_beats)
        session.state.completed_dialogue_beats = []  # Reset for new scene
        logger.debug(f"  ✓ State updated - Reset {completed_beats_before} completed beats for new scene")

        # Save to database
        logger.debug(f"  → Saving to database...")
        from sqlalchemy import update

        from letta.orm.story import StorySession as StorySessionORM

        async with db_registry.async_session() as db_session:
            async with db_session.begin():
                stmt = update(StorySessionORM).where(StorySessionORM.session_id == session_id).values(state=session.state.dict())
                result = await db_session.execute(stmt)
                logger.debug(f"  ✓ Database updated - Rows affected: {result.rowcount}")

        logger.info(f"✅ Q6 Scene Advance SUCCESS - {previous_scene} → {new_scene_num} ('{new_scene.title}')")

        # Update NPC memory blocks with new scene context
        try:
            update_result = await session_manager.update_scene_memory_blocks(
                session_id=session_id,
                new_scene_number=new_scene_num,
                actor=actor,
            )
            logger.info(
                f"  🧠 Updated {update_result['updated_agents']}/{update_result['total_agents']} "
                f"agent memories for Scene {new_scene_num}"
            )
        except Exception as e:
            # Don't fail the entire scene transition if memory update fails
            logger.error(f"  ⚠️ Failed to update scene memories: {e}", exc_info=True)

        return {
            "success": True,
            "previous_scene": previous_scene,
            "new_scene": new_scene_num,
            "new_scene_title": new_scene.title,
            "new_scene_location": new_scene.location,
            "message": f"Advanced to Scene {new_scene_num}: {new_scene.title}",
            "total_scenes": len(story.scenes),
        }

    except ValueError as e:
        logger.error(f"❌ Q6 Manual Scene Advance FAILED (ValueError): {e}")
        logger.debug(f"  Context - Session: {session_id}, User: {actor.id}")
        return JSONResponse(
            status_code=404,
            content={"error": "SCENE_ADVANCE_ERROR", "message": str(e), "suggestions": ["Verify session ID", "Check story exists"]},
        )

    except Exception as e:
        logger.error(f"❌ Q6 Manual Scene Advance FAILED (Exception): {e}", exc_info=True)
        logger.debug(f"  Context - Session: {session_id}, User: {actor.id}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "SCENE_ADVANCE_ERROR",
                "message": f"Failed to advance scene: {str(e)}",
                "suggestions": ["Check server logs", "Try again", "Contact support if issue persists"],
            },
        )


@router.post("/sessions/{session_id}/skip-beat")
async def skip_beat_manually(
    session_id: str,
    beat_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Q7: Manually skip/override a beat if story gets stuck.

    **Purpose:**
    This endpoint allows Unity or QA to force-complete a beat when:
    - Beat detection fails (keywords don't match)
    - Story gets stuck on a beat
    - QA testing needs to skip content
    - Player explicitly wants to skip

    **How It Works:**
    1. Gets current session and story
    2. Validates beat exists and is pending
    3. Marks beat as completed in session state
    4. Updates database
    5. Returns updated session state

    **Use Cases:**
    - Beat keyword matching failed
    - Dialogue went off-script but covered topic
    - QA testing / debugging
    - Player skip button

    **Example Request:**
    ```
    POST /v1/story/sessions/session-abc-123/skip-beat?beat_id=scene-1-beat-2
    ```

    **Example Response:**
    ```json
    {
        "success": true,
        "beat_id": "scene-1-beat-2",
        "beat_type": "dialogue",
        "message": "Beat skipped: scene-1-beat-2 (dialogue)",
        "new_progress": 0.66,
        "scene_complete": false
    }
    ```

    **Returns:**
    - Success status
    - Skipped beat ID and type
    - Updated progress
    - Scene completion status

    **Errors:**
    - 404: Session not found
    - 404: Beat not found
    - 400: Beat already completed
    """
    from letta.services.session_manager import SessionManager
    from letta.services.story_manager import StoryManager

    actor = await server.user_manager.get_actor_or_default_async(actor_id=actor_id)
    logger.info(f"📍 Q7 Beat Skip Request - Session: {session_id}, Beat: {beat_id}, User: {actor.id}")

    try:
        session_manager = SessionManager()
        story_manager = StoryManager()
        logger.debug(f"  ✓ Managers initialized")

        # Get session
        session = await session_manager._get_session_by_id(session_id, actor)
        if not session:
            logger.warning(f"  ✗ Session not found: {session_id}")
            raise ValueError(f"Session '{session_id}' not found")
        logger.debug(f"  ✓ Session retrieved - Scene: {session.state.current_scene_number}")

        # Get story
        story = await story_manager.get_story(session.story_id, actor)
        if not story:
            logger.warning(f"  ✗ Story not found: {session.story_id}")
            raise ValueError(f"Story '{session.story_id}' not found")
        logger.debug(f"  ✓ Story retrieved - '{story.title}'")

        # Find beat in current scene
        current_scene_num = session.state.current_scene_number
        current_scene = story.scenes[current_scene_num - 1]

        beat_type = None
        beat_found = None

        # Check dialogue beats
        for beat in current_scene.dialogue_beats:
            if beat.get("beat_id") == beat_id:
                beat_found = beat
                beat_type = "dialogue"
                break

        # Q5: Check narration beats
        if not beat_found:
            for beat in current_scene.narration_beats:
                if beat.get("beat_id") == beat_id:
                    beat_found = beat
                    beat_type = "narration"
                    break

        # Q5: Check action beats
        if not beat_found:
            for beat in current_scene.action_beats:
                if beat.get("beat_id") == beat_id:
                    beat_found = beat
                    beat_type = "action"
                    break

        if not beat_found:
            logger.error(f"  ✗ Beat not found: {beat_id}")
            return JSONResponse(
                status_code=404,
                content={
                    "error": "BEAT_NOT_FOUND",
                    "message": f"Beat '{beat_id}' not found in current scene",
                    "suggestions": ["Check beat ID", "Verify you're in the correct scene"],
                },
            )

        logger.debug(f"  ✓ Beat found - Type: {beat_type}, Priority: {beat_found.get('priority', 'required')}")

        # Check if already completed
        completed_list = {
            "dialogue": session.state.completed_dialogue_beats,
            "narration": session.state.completed_narration_beats,
            "action": session.state.completed_action_beats,
        }[beat_type]

        if beat_id in completed_list:
            logger.warning(f"  ⚠️ Beat already completed: {beat_id}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "BEAT_ALREADY_COMPLETED",
                    "message": f"Beat '{beat_id}' is already completed",
                    "suggestions": ["Check session state", "Use different beat ID"],
                },
            )

        # Mark beat as completed
        logger.info(f"  ⏭️ Skipping beat: {beat_id} ({beat_type})")
        completed_list.append(beat_id)

        # Calculate new progress
        from letta.services.dialogue_manager import DialogueManager

        dialogue_manager = DialogueManager()
        scene_complete, progress = dialogue_manager._check_scene_completion(current_scene, session.state.completed_dialogue_beats)

        logger.debug(f"  ✓ Beat marked complete - Progress: {progress:.2%}, Scene complete: {scene_complete}")

        # Save to database (Q7 FIX: Fetch and modify ORM object directly)
        logger.debug(f"  → Saving to database...")
        logger.debug(
            f"  → State before save: D={len(session.state.completed_dialogue_beats)}, N={len(session.state.completed_narration_beats)}, A={len(session.state.completed_action_beats)}"
        )
        from sqlalchemy import select, update
        from sqlalchemy.orm import selectinload

        from letta.orm.story import StorySession as StorySessionORM

        # Serialize state properly for JSON storage
        state_dict = session.state.model_dump(mode="json") if hasattr(session.state, "model_dump") else session.state.dict()
        logger.debug(f"  → Narration beats in serialized state: {state_dict.get('completed_narration_beats', [])}")

        async with db_registry.async_session() as db_session:
            async with db_session.begin():
                # Fetch the ORM object
                stmt = select(StorySessionORM).where(StorySessionORM.session_id == session_id)
                result = await db_session.execute(stmt)
                session_orm = result.scalar_one()

                # Update the state directly
                session_orm.state = state_dict

                # SQLAlchemy will auto-commit on context exit
                logger.debug(f"  ✓ Database updated via ORM object modification")

        logger.info(f"✅ Q7 Beat Skip SUCCESS - {beat_id} ({beat_type}) - Progress: {progress:.2%}")

        return {
            "success": True,
            "beat_id": beat_id,
            "beat_type": beat_type,
            "message": f"Beat skipped: {beat_id} ({beat_type})",
            "new_progress": round(progress, 3),
            "scene_complete": scene_complete,
            "total_completed": {
                "dialogue": len(session.state.completed_dialogue_beats),
                "narration": len(session.state.completed_narration_beats),
                "action": len(session.state.completed_action_beats),
            },
        }

    except ValueError as e:
        logger.error(f"❌ Q7 Beat Skip FAILED (ValueError): {e}")
        logger.debug(f"  Context - Session: {session_id}, Beat: {beat_id}, User: {actor.id}")
        return JSONResponse(
            status_code=404,
            content={
                "error": "BEAT_SKIP_ERROR",
                "message": str(e),
                "suggestions": ["Verify session ID", "Check story exists", "Verify beat ID"],
            },
        )

    except Exception as e:
        logger.error(f"❌ Q7 Beat Skip FAILED (Exception): {e}", exc_info=True)
        logger.debug(f"  Context - Session: {session_id}, Beat: {beat_id}, User: {actor.id}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "BEAT_SKIP_ERROR",
                "message": f"Failed to skip beat: {str(e)}",
                "suggestions": ["Check server logs", "Try again", "Contact support if issue persists"],
            },
        )


# ============================================================
# GET Endpoints for Unity Integration (Kon's Requirements)
# ============================================================


@router.get("/{story_id}", response_model=StoryDetailResponse)
async def get_story_details(
    story_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Get full story structure for caching.

    **Purpose:**
    This endpoint provides the complete story structure (all scenes, characters,
    dialogue beats, and instructions) so Kon's Unity server can cache this data
    locally and avoid repeated queries.

    **Use Case:**
    - When Kon's server first loads a story
    - When Unity client needs story metadata
    - For displaying story overview/map

    **Example Request:**
    ```
    GET /api/v1/story/story-1001
    ```

    **Example Response:**
    ```json
    {
        "story_id": "story-1001",
        "title": "The Price of Survival",
        "description": "A survival horror story...",
        "characters": [...],
        "player_character": {...},
        "npcs": [...],
        "scenes": [
            {
                "scene_id": "scene-1",
                "scene_number": 1,
                "title": "The Nightmare Begins",
                "location": "Abandoned Factory",
                "dialogue_beats": [...]
            }
        ],
        "total_scenes": 5,
        "metadata": {}
    }
    ```

    **Returns:**
    - Full story structure
    - All characters (player + NPCs)
    - All scenes with dialogue beats
    - Story metadata

    **Errors:**
    - 404: Story not found
    """
    from letta.schemas.story import StoryDetailResponse

    actor = await server.user_manager.get_actor_or_default_async(actor_id=actor_id)
    logger.info(f"📖 Story details request: {story_id} (user: {actor.id})")

    try:
        session_manager = SessionManager()
        response = await session_manager.get_story_details(story_id, actor)

        logger.info(f"✅ Story details retrieved: {story_id}")
        return response

    except ValueError as e:
        logger.error(f"❌ Story not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "STORY_NOT_FOUND",
                "message": str(e),
                "suggestions": [
                    f"Verify story ID '{story_id}' exists",
                    "Check if you have permission to access this story",
                    "Upload the story first with POST /api/v1/story/upload",
                ],
            },
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Error retrieving story: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "STORY_RETRIEVAL_FAILED",
                "message": f"Failed to retrieve story: {error_msg}",
            },
        )


@router.get("/sessions/{session_id}/validate")
async def validate_session(
    session_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Validate if a session still exists and is active.
    
    **Purpose:**
    This endpoint helps clients check if a session ID is still valid.
    Sessions can become invalid if:
    - The story was re-uploaded with ?overwrite=true
    - The session was explicitly deleted
    - The session expired
    
    **Use Case:**
    - After story upload with overwrite=true
    - Before resuming a cached session
    - For error recovery
    
    **Example Request:**
    ```
    GET /api/v1/story/sessions/session-abc-123/validate
    ```
    
    **Example Response (Valid):**
    ```json
    {
        "valid": true,
        "session_id": "session-abc-123",
        "story_id": "story-1001",
        "status": "active",
        "created_at": "2025-11-13T10:00:00Z"
    }
    ```
    
    **Example Response (Invalid):**
    ```json
    {
        "valid": false,
        "session_id": "session-abc-123",
        "reason": "Session not found (may have been deleted during story overwrite)",
        "suggestions": [
            "Start a new session with POST /v1/story/sessions/start",
            "Do not reuse old session IDs after story overwrite"
        ]
    }
    ```
    
    **Returns:**
    - Validation status
    - Session metadata if valid
    - Reason and suggestions if invalid
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=actor_id)
    logger.info(f"🔍 Session validation request: {session_id} (user: {actor.id})")
    
    try:
        session_manager = SessionManager()
        session = await session_manager._get_session_by_id(session_id, actor)
        
        if not session:
            logger.warning(f"  ❌ Session not found: {session_id}")
            return JSONResponse(
                status_code=200,  # 200 for validation endpoint, not 404
                content={
                    "valid": False,
                    "session_id": session_id,
                    "reason": "Session not found (may have been deleted during story overwrite)",
                    "suggestions": [
                        "Start a new session with POST /v1/story/sessions/start",
                        "Do not reuse old session IDs after story overwrite",
                        "Check if story was re-uploaded with ?overwrite=true",
                    ],
                },
            )
        
        logger.info(f"  ✅ Session is valid: {session_id}")
        return {
            "valid": True,
            "session_id": session.session_id,
            "story_id": session.story_id,
            "status": session.status,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        }
    
    except Exception as e:
        logger.error(f"❌ Session validation error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "valid": False,
                "session_id": session_id,
                "reason": f"Validation error: {str(e)}",
                "suggestions": ["Check server logs", "Try again"],
            },
        )


@router.get("/sessions/{session_id}/messages/recent")
async def get_recent_messages(
    session_id: str,
    limit: int = Query(10, description="Number of recent messages to retrieve (default: 10, max: 50)"),
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Get recent dialogue messages for a session.
    
    Returns the last N messages from all characters in chronological order,
    including both player and NPC messages.
    
    **Use Case:**
    - Show context when player reopens a story
    - Build chat UI replay
    - Debug dialogue flow
    - Display conversation history in Unity
    
    **Example Request:**
    ```
    GET /api/v1/story/sessions/session-abc-123/messages/recent?limit=5
    ```
    
    **Example Response:**
    ```json
    [
        {
            "character": "Alex",
            "message": "Hey! I've been thinking about what you said earlier.",
            "timestamp": "2025-11-17T14:29:45Z",
            "role": "assistant"
        },
        {
            "character": "player",
            "message": "What do you think we should do?",
            "timestamp": "2025-11-17T14:30:02Z",
            "role": "user"
        },
        {
            "character": "Alex",
            "message": "We need to find a way out. The back door might be our best bet.",
            "timestamp": "2025-11-17T14:30:15Z",
            "role": "assistant"
        }
    ]
    ```
    
    **Returns:**
    - List of recent messages with character name, message text, timestamp, and role
    - Messages sorted chronologically (oldest first)
    - Limited to requested number (default 10, max 50)
    
    **Errors:**
    - 404: Session not found
    - 500: Database error
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=actor_id)
    logger.info(f"📜 Recent messages request: {session_id}, limit={limit} (user: {actor.id})")
    
    try:
        # Validate limit
        if limit < 1:
            limit = 10
        elif limit > 50:
            limit = 50
        
        session_manager = SessionManager()
        messages = await session_manager.get_recent_messages(
            session_id=session_id,
            limit=limit,
            actor=actor,
            server=server,
        )
        
        logger.info(f"✅ Retrieved {len(messages)} recent messages for session: {session_id}")
        return messages
    
    except ValueError as e:
        logger.error(f"❌ Session not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "SESSION_NOT_FOUND",
                "message": str(e),
                "suggestions": [
                    f"Session '{session_id}' not found",
                    "Verify session ID is correct",
                    "Session may have been deleted",
                ],
            },
        )
    
    except Exception as e:
        logger.error(f"❌ Failed to retrieve recent messages: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "MESSAGES_RETRIEVAL_FAILED",
                "message": f"Failed to retrieve messages: {str(e)}",
                "suggestions": [
                    "Check server logs for details",
                    "Verify database connection",
                    "Try again in a moment",
                ],
            },
        )


@router.get("/sessions/{session_id}/state", response_model=SessionStateResponse)
async def get_session_state(
    session_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Get comprehensive session state for Unity.

    **Purpose:**
    This is the MAIN endpoint that Kon's server will call to get everything Unity needs:
    - Current scene/setting
    - Available characters
    - Next instruction/beat
    - Story progress

    **Use Case:**
    - When Unity loads a session
    - When checking what instruction comes next
    - For displaying progress UI
    - For determining which characters are available

    **Example Request:**
    ```
    GET /api/v1/story/sessions/session-abc-123/state
    ```

    **Example Response:**
    ```json
    {
        "story_id": "story-1001",
        "story_title": "The Price of Survival",
        "session_id": "session-abc-123",
        "session_status": "active",
        "current_setting": {
            "scene_id": "scene-1",
            "scene_number": 1,
            "scene_title": "The Nightmare Begins",
            "location": "Abandoned Factory - Main Hall",
            "total_scenes": 5
        },
        "player_character": "Woo",
        "available_npcs": [
            {
                "character_id": "char-002",
                "name": "Ah-rin",
                "age": 14,
                "sex": "female",
                "model": "young_girl_casual",
                "role": "ah-rin"
            },
            {
                "character_id": "char-003",
                "name": "Ji-woo",
                "age": 17,
                "sex": "male",
                "model": "teen_boy_hoodie",
                "role": "ji-woo"
            }
        ],
        "next_instruction": {
            "type": "dialogue",
            "beat_id": "scene-1-beat-2",
            "beat_number": 2,
            "character": "Ah-rin",
            "topic": "escape plans",
            "script_text": "We need to find a way out of this nightmare",
            "is_completed": false,
            "instruction_details": {
                "general_guidance": "Guide conversation toward: escape plans",
                "emotional_tone": "natural",
                "keywords": ["escape", "plans", "nightmare", "need", "find"]
            }
        },
        "progress": {
            "scene_progress": 0.33,
            "beats_completed": ["scene-1-beat-1"],
            "beats_remaining": ["scene-1-beat-2", "scene-1-beat-3"],
            "total_beats_in_scene": 3,
            "scene_complete": false
        },
        "metadata": {
            "last_updated": "2025-10-28T10:30:00Z",
            "total_interactions": 1,
            "current_instruction_index": 1
        }
    }
    ```

    **Returns:**
    - Current scene info (setting, location, title)
    - Player character name
    - Available NPCs for interaction
    - Next instruction/beat to address
    - Progress through current scene
    - Session metadata

    **Errors:**
    - 404: Session not found
    """
    from letta.schemas.story import SessionStateResponse

    actor = await server.user_manager.get_actor_or_default_async(actor_id=actor_id)
    logger.info(f"🎮 Session state request: {session_id} (user: {actor.id})")

    try:
        session_manager = SessionManager()
        response = await session_manager.get_session_state(session_id, actor)

        logger.info(f"✅ Session state retrieved: {session_id}")
        return response

    except ValueError as e:
        logger.error(f"❌ Session not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "SESSION_NOT_FOUND",
                "message": str(e),
                "suggestions": [
                    f"Verify session ID '{session_id}' exists",
                    "Check if session is still active",
                    "Start a new session with POST /api/v1/story/sessions/start",
                ],
            },
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Error retrieving session state: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "SESSION_STATE_RETRIEVAL_FAILED",
                "message": f"Failed to retrieve session state: {error_msg}",
            },
        )


@router.get("/sessions/{session_id}/relationships", response_model=RelationshipStatusResponse)
async def get_relationship_status(
    session_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Get current relationship status for a session.
    
    **Purpose:**
    Lightweight endpoint for Kon to quickly query relationship status without
    fetching the entire session state. Perfect for:
    - Updating relationship meters/bars in Unity UI
    - Checking relationship thresholds before showing choices
    - Polling for relationship changes after dialogue
    
    **Use Cases:**
    - Before showing choices: Check if player has sufficient relationship level
    - After dialogue: Update UI with new relationship values
    - Real-time UI: Poll this endpoint to keep meters updated
    
    **Example Request:**
    ```
    GET /api/v1/story/sessions/session-abc-123/relationships
    Headers:
      user_id: your-user-id
    ```
    
    **Example Response:**
    ```json
    {
        "session_id": "session-abc-123",
        "relationship_points": {
            "tatsuya-friendship": 130,
            "tatsuya-romance": 25,
            "rina-friendship": 60
        },
        "relationship_levels": {
            "tatsuya-friendship": 1,
            "tatsuya-romance": 0,
            "rina-friendship": 0
        },
        "relationships_defined": [
            "tatsuya-friendship",
            "tatsuya-romance",
            "rina-friendship"
        ],
        "timestamp": "2025-11-13T14:30:00Z"
    }
    ```
    
    **Returns:**
    - Current relationship points for all relationships
    - Current relationship levels for all relationships
    - List of defined relationships in the story
    - Timestamp of query
    
    **Performance:**
    - Very fast: Only fetches session state (no story structure)
    - Lightweight: Returns only relationship data
    - Can be polled frequently without performance impact
    
    **Errors:**
    - 404: Session not found
    - 500: Database error
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=actor_id)
    logger.info(f"💝 Relationship status request: session={session_id} (user: {actor.id})")
    
    try:
        session_manager = SessionManager()
        
        # Get session
        session = await session_manager._get_session_by_id(session_id, actor)
        if not session:
            logger.error(f"❌ Session not found: {session_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "SESSION_NOT_FOUND",
                    "message": f"Session '{session_id}' not found",
                    "suggestions": [
                        "Verify session ID is correct",
                        "Check if session is still active",
                        "Start a new session with POST /api/v1/story/sessions/start",
                    ],
                },
            )
        
        # Get story to find defined relationships
        story = await session_manager.story_manager.get_story(session.story_id, actor)
        if not story:
            logger.error(f"❌ Story not found: {session.story_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "STORY_NOT_FOUND",
                    "message": f"Story '{session.story_id}' not found",
                },
            )
        
        # Extract relationship data from session state
        # SessionState already has arrays, just pass them through
        relationship_points = session.state.relationship_points or []
        relationship_levels = session.state.relationship_levels or []
        
        # Get list of defined relationships from story
        relationships_defined = []
        if story.relationships:
            relationships_defined = [rel.relationship_id for rel in story.relationships if rel.relationship_id]
        
        logger.info(
            f"✅ Relationship status retrieved: {session_id} "
            f"({len(relationship_points)} relationships, {len(relationships_defined)} defined)"
        )
        
        return RelationshipStatusResponse(
            session_id=session.session_id,
            relationship_points=relationship_points,
            relationship_levels=relationship_levels,
            relationships_defined=relationships_defined,
        )
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Error retrieving relationship status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "RELATIONSHIP_STATUS_RETRIEVAL_FAILED",
                "message": f"Failed to retrieve relationship status: {error_msg}",
                "suggestions": [
                    "Check server logs for details",
                    "Verify session ID is valid",
                    "Try again in a moment",
                ],
            },
        )


# ============================================================
# Health Check
# ============================================================


@router.get("/health")
async def story_health():
    """
    Health check for story system.
    
    Returns system status and available features.
    """
    return {
        "status": "healthy",
        "features": {
            "story_upload": "enabled",
            "session_management": "enabled",
            "dialogue_generation": "enabled",  # ✅ NOW ENABLED!
        },
        "version": "1.0.0",
        "dialogue_config": {
            "script_strictness": "contextual",  # Q6: Option C
            "off_topic_handling": "brief_redirect",  # Q7: Option B
            "scene_progression": "automatic",  # Q8: Option A
            "allow_skip": False,  # Q9: No
            "error_fallback": "scripted_text_with_retry",  # Q10: Scripted + Retry
        },
    }
