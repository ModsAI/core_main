"""
Story System API Router

Provides endpoints for:
- Story upload and retrieval
- Session management (start, resume, restart, delete)
- Dialogue generation (coming soon after Q6-Q10 answered)
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from letta.log import get_logger
from letta.schemas.story import (
    SessionCreate,
    SessionResume,
    SessionRestartResponse,
    SessionStartResponse,
    StoryError,
    StoryUpload,
    StoryUploadResponse,
)
from letta.schemas.user import User
from letta.server.rest_api.dependencies import get_current_user
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
    user: User = Depends(get_current_user),
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
    logger.info(f"📤 Story upload request: {story.title} (user: {user.id})")
    
    try:
        story_manager = StoryManager()
        response = await story_manager.upload_story(story, user)
        
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
    user: User = Depends(get_current_user),
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
    logger.info(f"🎬 Session start request: {session_create.story_id} (user: {user.id})")
    
    try:
        session_manager = SessionManager()
        response = await session_manager.start_session(session_create, user)
        
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


@router.get("/sessions/{session_id}/resume", response_model=SessionResume)
async def resume_session(
    session_id: str,
    user: User = Depends(get_current_user),
):
    """
    Resume an existing session.
    
    Returns current session state, current scene, and recent interaction history.
    
    **Example:**
    ```
    GET /api/v1/story/sessions/session-abc123/resume
    ```
    
    **Returns:**
    - Session details
    - Current scene
    - Current progress
    - Recent chat history
    
    **Errors:**
    - 404: Session not found
    - 500: Database error
    """
    logger.info(f"▶️ Session resume request: {session_id} (user: {user.id})")
    
    try:
        session_manager = SessionManager()
        response = await session_manager.resume_session(session_id, user)
        
        logger.info(f"✅ Session resumed: {session_id}")
        return response
    
    except ValueError as e:
        logger.error(f"❌ Session resume validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "SESSION_NOT_FOUND",
                "message": str(e),
                "suggestions": [
                    f"Verify session ID '{session_id}' exists",
                    "Check if session belongs to your user",
                    "Session may have been deleted",
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
    user: User = Depends(get_current_user),
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
    logger.info(f"🔄 Session restart request: {session_id} (user: {user.id})")
    
    try:
        session_manager = SessionManager()
        response = await session_manager.restart_session(session_id, user)
        
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
    user: User = Depends(get_current_user),
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
    logger.info(f"🗑️ Session delete request: {session_id} (user: {user.id})")
    
    try:
        session_manager = SessionManager()
        deleted = await session_manager.delete_session(session_id, user)
        
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
    user: User = Depends(get_current_user),
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
    logger.info(f"💬 Dialogue request: session={session_id}, character={request.target_character}")
    
    try:
        from letta.services.dialogue_manager import DialogueManager
        
        dialogue_manager = DialogueManager()
        response = await dialogue_manager.generate_dialogue(session_id, request, user)
        
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

