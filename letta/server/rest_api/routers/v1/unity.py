"""
Unity Integration API Router

Provides REST endpoints for Unity game integration with Letta agents.
Follows the same patterns as existing Letta API routers.
"""

from typing import List, Optional

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query  # Header for actor_id auth
from fastapi.responses import JSONResponse

from letta.log import get_logger
from letta.orm.errors import NoResultFound, UniqueConstraintViolationError
from letta.otel.tracing import trace_method
from letta.schemas.unity_character import (
    CharacterTier,
    UnityCharacter,
    UnityCharacterCreate,
    UnityCharacterError,
    UnityCharacterListResponse,
    UnityCharacterRegistrationResponse,
    UnityCharacterUpdate,
    UnityDialogueRequest,
    UnityDialogueResponse,
)
from letta.server.rest_api.utils import get_letta_server
from letta.server.server import SyncServer
from letta.services.unity_character_manager import UnityCharacterManager

logger = get_logger(__name__)

router = APIRouter(prefix="/unity", tags=["unity"])


@router.post(
    "/characters/register",
    response_model=UnityCharacterRegistrationResponse,
    operation_id="register_unity_character",
)
@trace_method
async def register_unity_character(
    request: UnityCharacterCreate = Body(...),
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Register a new Unity character and create its associated Letta agent.
    
    This endpoint allows the Unity team to register characters for AI-powered dialogue.
    Each character gets a dedicated Letta agent with the specified personality and backstory.
    
    Args:
        request: Character registration data
        server: Letta server instance
        actor: Current authenticated user
        
    Returns:
        Registration response with character details and dialogue endpoint
        
    Raises:
        409: If unity_character_id already exists
        400: If validation fails
        500: If agent creation fails
    """
    logger.info(
        f"🎭 Unity character registration request: {request.character_name} "
        f"(ID: {request.unity_character_id}, Game: {request.game_id})"
    )

    try:
        # Use the Unity Character Manager to create the character
        character_manager = UnityCharacterManager()
        character = await character_manager.create_unity_character(request, actor)

        # Build response with next steps for Unity team
        instructions = [
            f"Character '{request.character_name}' successfully registered",
            f"Use dialogue endpoint: POST /api/v1/unity/characters/{request.unity_character_id}/dialogue",
            "Send player messages to get AI-powered responses",
            "Character will remember all interactions automatically",
        ]

        response = UnityCharacterRegistrationResponse(
            success=True,
            character=character,
            letta_agent_id=character.letta_agent_id,
            dialogue_endpoint=f"/api/v1/unity/characters/{request.unity_character_id}/dialogue",
            instructions=instructions,
        )

        logger.info(f"✅ Successfully registered Unity character: {request.character_name}")
        return response

    except UniqueConstraintViolationError as e:
        logger.error(f"❌ Unity character ID conflict: {e}")
        error_response = UnityCharacterError(
            error_code="DUPLICATE_CHARACTER_ID",
            error_message=f"Unity character ID '{request.unity_character_id}' already exists",
            details={"unity_character_id": request.unity_character_id},
            suggestions=[
                "Use a different unity_character_id",
                "Check if character was already registered",
                "Use GET /api/v1/unity/characters to list existing characters",
            ],
        )
        raise HTTPException(status_code=409, detail=error_response.dict())

    except ValueError as e:
        logger.error(f"❌ Unity character validation error: {e}")
        error_response = UnityCharacterError(
            error_code="VALIDATION_ERROR",
            error_message=str(e),
            suggestions=[
                "Check all required fields are provided",
                "Ensure field lengths are within limits",
                "Verify unity_character_id format",
            ],
        )
        raise HTTPException(status_code=400, detail=error_response.dict())

    except Exception as e:
        logger.error(f"❌ Unity character registration failed: {e}", exc_info=True)
        error_response = UnityCharacterError(
            error_code="REGISTRATION_FAILED",
            error_message="Failed to register Unity character",
            details={"error": str(e)},
            suggestions=[
                "Check server logs for detailed error information",
                "Ensure Letta agent creation is working",
                "Retry the request",
            ],
        )
        raise HTTPException(status_code=500, detail=error_response.dict())


@router.get(
    "/characters",
    response_model=UnityCharacterListResponse,
    operation_id="list_unity_characters",
)
@trace_method
async def list_unity_characters(
    game_id: Optional[str] = Query(None, description="Filter by game ID"),
    character_tier: Optional[CharacterTier] = Query(None, description="Filter by character tier"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    limit: int = Query(50, description="Maximum number of results", ge=1, le=100),
    offset: int = Query(0, description="Pagination offset", ge=0),
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    List Unity characters with optional filters.
    
    Args:
        game_id: Optional game ID filter
        character_tier: Optional character tier filter
        is_active: Optional active status filter  
        limit: Maximum results (1-100)
        offset: Pagination offset
        server: Letta server instance
        actor: Current authenticated user
        
    Returns:
        List of Unity characters matching the filters
    """
    logger.info(f"📋 Listing Unity characters (game_id={game_id}, tier={character_tier}, active={is_active})")

    try:
        character_manager = UnityCharacterManager()
        characters = await character_manager.list_unity_characters(
            actor=actor,
            game_id=game_id,
            character_tier=character_tier,
            is_active=is_active,
            limit=limit,
            offset=offset,
        )

        response = UnityCharacterListResponse(
            characters=characters,
            total=len(characters),  # TODO: Get actual total count
            game_id=game_id,
        )

        logger.info(f"✅ Found {len(characters)} Unity characters")
        return response

    except Exception as e:
        logger.error(f"❌ Failed to list Unity characters: {e}", exc_info=True)
        error_response = UnityCharacterError(
            error_code="LIST_FAILED",
            error_message="Failed to list Unity characters",
            details={"error": str(e)},
        )
        raise HTTPException(status_code=500, detail=error_response.dict())


@router.get(
    "/characters/{unity_character_id}",
    response_model=UnityCharacter,
    operation_id="get_unity_character",
)
@trace_method
async def get_unity_character(
    unity_character_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Get a specific Unity character by ID.
    
    Args:
        unity_character_id: The Unity character ID
        server: Letta server instance
        actor: Current authenticated user
        
    Returns:
        Unity character details
        
    Raises:
        404: If character not found
    """
    logger.info(f"🔍 Getting Unity character: {unity_character_id}")

    try:
        character_manager = UnityCharacterManager()
        character = await character_manager.get_unity_character_by_id(unity_character_id, actor)

        if not character:
            logger.warning(f"❌ Unity character not found: {unity_character_id}")
            error_response = UnityCharacterError(
                error_code="CHARACTER_NOT_FOUND",
                error_message=f"Unity character '{unity_character_id}' not found",
                suggestions=[
                    "Check the unity_character_id is correct",
                    "Use GET /api/v1/unity/characters to list available characters",
                ],
            )
            raise HTTPException(status_code=404, detail=error_response.dict())

        logger.info(f"✅ Found Unity character: {character.character_name}")
        return character

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get Unity character: {e}", exc_info=True)
        error_response = UnityCharacterError(
            error_code="GET_CHARACTER_FAILED",
            error_message="Failed to get Unity character",
            details={"error": str(e)},
        )
        raise HTTPException(status_code=500, detail=error_response.dict())


@router.put(
    "/characters/{unity_character_id}",
    response_model=UnityCharacter,
    operation_id="update_unity_character",
)
@trace_method
async def update_unity_character(
    unity_character_id: str,
    request: UnityCharacterUpdate = Body(...),
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Update an existing Unity character.
    
    Args:
        unity_character_id: The Unity character ID
        request: Update data
        server: Letta server instance
        actor: Current authenticated user
        
    Returns:
        Updated Unity character
        
    Raises:
        404: If character not found
        400: If validation fails
    """
    logger.info(f"📝 Updating Unity character: {unity_character_id}")

    try:
        character_manager = UnityCharacterManager()
        character = await character_manager.update_unity_character(unity_character_id, request, actor)

        if not character:
            logger.warning(f"❌ Unity character not found for update: {unity_character_id}")
            error_response = UnityCharacterError(
                error_code="CHARACTER_NOT_FOUND",
                error_message=f"Unity character '{unity_character_id}' not found",
                suggestions=[
                    "Check the unity_character_id is correct",
                    "Use GET /api/v1/unity/characters to list available characters",
                ],
            )
            raise HTTPException(status_code=404, detail=error_response.dict())

        logger.info(f"✅ Updated Unity character: {character.character_name}")
        return character

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"❌ Unity character update validation error: {e}")
        error_response = UnityCharacterError(
            error_code="VALIDATION_ERROR",
            error_message=str(e),
            suggestions=[
                "Check field values are within allowed ranges",
                "Ensure required fields are not empty",
            ],
        )
        raise HTTPException(status_code=400, detail=error_response.dict())
    except Exception as e:
        logger.error(f"❌ Failed to update Unity character: {e}", exc_info=True)
        error_response = UnityCharacterError(
            error_code="UPDATE_FAILED",
            error_message="Failed to update Unity character",
            details={"error": str(e)},
        )
        raise HTTPException(status_code=500, detail=error_response.dict())


@router.delete(
    "/characters/{unity_character_id}",
    operation_id="delete_unity_character",
)
@trace_method
async def delete_unity_character(
    unity_character_id: str,
    cleanup_agent: bool = Query(True, description="Whether to delete the associated Letta agent"),
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Delete a Unity character and optionally its associated Letta agent.
    
    Args:
        unity_character_id: The Unity character ID
        cleanup_agent: Whether to delete the associated Letta agent
        server: Letta server instance
        actor: Current authenticated user
        
    Returns:
        Success confirmation
        
    Raises:
        404: If character not found
    """
    logger.info(f"🗑️ Deleting Unity character: {unity_character_id} (cleanup_agent={cleanup_agent})")

    try:
        character_manager = UnityCharacterManager()
        deleted = await character_manager.delete_unity_character(unity_character_id, actor, cleanup_agent)

        if not deleted:
            logger.warning(f"❌ Unity character not found for deletion: {unity_character_id}")
            error_response = UnityCharacterError(
                error_code="CHARACTER_NOT_FOUND",
                error_message=f"Unity character '{unity_character_id}' not found",
                suggestions=[
                    "Check the unity_character_id is correct",
                    "Character may have already been deleted",
                ],
            )
            raise HTTPException(status_code=404, detail=error_response.dict())

        logger.info(f"✅ Successfully deleted Unity character: {unity_character_id}")
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Unity character '{unity_character_id}' deleted successfully",
                "agent_cleaned": cleanup_agent,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete Unity character: {e}", exc_info=True)
        error_response = UnityCharacterError(
            error_code="DELETE_FAILED",
            error_message="Failed to delete Unity character",
            details={"error": str(e)},
        )
        raise HTTPException(status_code=500, detail=error_response.dict())


@router.post(
    "/characters/{unity_character_id}/dialogue",
    response_model=UnityDialogueResponse,
    operation_id="get_character_dialogue",
)
@trace_method
async def get_character_dialogue(
    unity_character_id: str,
    request: UnityDialogueRequest = Body(...),
    server: "SyncServer" = Depends(get_letta_server),
    actor_id: str | None = Header(None, alias="user_id"),
):
    """
    Get AI-powered dialogue response from a Unity character.
    
    This is the main endpoint Unity will use to get character responses.
    The character's Letta agent processes the player message and returns
    a contextual response with emotion and animation suggestions.
    
    Args:
        unity_character_id: The Unity character ID
        request: Dialogue request with player message and context
        server: Letta server instance
        actor: Current authenticated user
        
    Returns:
        Character dialogue response with text, emotion, and suggestions
        
    Raises:
        404: If character not found
        400: If request validation fails
        500: If dialogue generation fails
    """
    logger.info(f"💬 Dialogue request for character {unity_character_id}: '{request.player_message[:50]}...'")

    try:
        # Get the character
        character_manager = UnityCharacterManager()
        character = await character_manager.get_unity_character_by_id(unity_character_id, actor)

        if not character:
            logger.warning(f"❌ Unity character not found: {unity_character_id}")
            error_response = UnityCharacterError(
                error_code="CHARACTER_NOT_FOUND",
                error_message=f"Unity character '{unity_character_id}' not found",
                suggestions=[
                    "Check the unity_character_id is correct",
                    "Ensure character was properly registered",
                ],
            )
            raise HTTPException(status_code=404, detail=error_response.dict())

        if not character.is_active:
            logger.warning(f"❌ Unity character is inactive: {unity_character_id}")
            error_response = UnityCharacterError(
                error_code="CHARACTER_INACTIVE",
                error_message=f"Unity character '{unity_character_id}' is not active",
                suggestions=[
                    "Activate the character first",
                    "Check character status with GET endpoint",
                ],
            )
            raise HTTPException(status_code=400, detail=error_response.dict())

        # TODO: Implement actual dialogue generation with Letta agent
        # For now, return a placeholder response
        logger.warning("⚠️ Dialogue generation not yet implemented - returning placeholder response")

        # Increment usage statistics
        await character_manager.increment_character_usage(unity_character_id, actor)

        # Placeholder response structure
        response = UnityDialogueResponse(
            character_id=unity_character_id,
            dialogue_text=f"[{character.character_name}] Hello! I heard you say: '{request.player_message}'. This is a placeholder response - full dialogue generation coming soon!",
            emotion="neutral",
            animation_suggestion="idle",
            memory_updated=True,
            relationship_change=1,
            next_action="wait_for_player_response",
        )

        logger.info(f"✅ Generated dialogue response for {character.character_name}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Dialogue generation failed: {e}", exc_info=True)
        error_response = UnityCharacterError(
            error_code="DIALOGUE_FAILED",
            error_message="Failed to generate character dialogue",
            details={"error": str(e)},
            suggestions=[
                "Check server logs for detailed error information",
                "Ensure Letta agent is functioning properly",
                "Retry the request",
            ],
        )
        raise HTTPException(status_code=500, detail=error_response.dict())


@router.get(
    "/health",
    operation_id="unity_health_check",
)
@trace_method
async def unity_health_check():
    """
    Health check endpoint for Unity integration.
    
    Returns:
        System status and version information
    """
    logger.debug("🏥 Unity health check requested")

    try:
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "service": "letta-unity-integration",
                "version": "1.0.0",
                "endpoints": {
                    "character_registration": "/api/v1/unity/characters/register",
                    "character_list": "/api/v1/unity/characters",
                    "character_dialogue": "/api/v1/unity/characters/{unity_character_id}/dialogue",
                },
                "timestamp": logger._get_utc_timestamp(),
            },
        )

    except Exception as e:
        logger.error(f"❌ Unity health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": logger._get_utc_timestamp(),
            },
        )


def _get_utc_timestamp() -> str:
    """Get current UTC timestamp as ISO string"""
    from datetime import datetime
    
    return datetime.utcnow().isoformat() + "Z"
