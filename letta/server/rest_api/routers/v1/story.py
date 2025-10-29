"""
Story System API Router

Provides endpoints for:
- Story upload and retrieval
- Session management (start, resume, restart, delete)
- Dialogue generation (coming soon after Q6-Q10 answered)
"""

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status

from letta.log import get_logger
from letta.schemas.story import (
    SessionCreate,
    SessionResume,
    SessionRestartResponse,
    SessionStartResponse,
    SessionStateResponse,
    StoryDetailResponse,
    StoryDialogueRequest,
    StoryDialogueResponse,
    StoryError,
    StoryUpload,
    StoryUploadResponse,
)
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
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Upload a new story.
    
    Converts TypeScript story JSON to internal format and stores in database.
    
    **Process:**
    1. Validates story structure
    2. Generates character IDs
    3. Parses instructions into scenes
    4. Extracts dialogue beats
    5. Stores in database
    
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
    logger.info(f"📤 Story upload request: {story.title} (user: {actor.id})")
    
    try:
        story_manager = StoryManager()
        response = await story_manager.upload_story(story, actor)
        
        logger.info(f"✅ Story uploaded: {response.story_id}")
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
                ]
            }
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
                    ]
                }
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
                ]
            }
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
                ]
            }
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
                ]
            }
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
        response = await session_manager.resume_session(session_resume.story_id, actor)
        
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
                ]
            }
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
                ]
            }
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
                ]
            }
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
                ]
            }
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
                }
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
                ]
            }
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
    logger.info(f"💬 Dialogue request: session={session_id}, character={request.target_character}")
    
    try:
        from letta.services.dialogue_manager import DialogueManager
        
        dialogue_manager = DialogueManager(server=server)
        response = await dialogue_manager.generate_dialogue(session_id, request, actor)
        
        logger.info(f"✅ Dialogue generated: {len(response.dialogue_text)} chars")
        return response
    
    except ValueError as e:
        logger.error(f"❌ Dialogue validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "DIALOGUE_ERROR",
                "message": str(e),
                "suggestions": [
                    "Verify session ID exists",
                    "Check character ID is valid",
                    "Ensure session is active",
                ]
            }
        )
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Dialogue generation error: {e}", exc_info=True)
        
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
                ]
            }
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
                ]
            }
        )
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Error retrieving story: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "STORY_RETRIEVAL_FAILED",
                "message": f"Failed to retrieve story: {error_msg}",
            }
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
                ]
            }
        )
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Error retrieving session state: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "SESSION_STATE_RETRIEVAL_FAILED",
                "message": f"Failed to retrieve session state: {error_msg}",
            }
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

