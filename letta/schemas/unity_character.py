"""
Unity Character Pydantic Schemas

Request/Response models for Unity character operations.
Follows the same patterns as existing Letta schemas (Agent, Block, etc.)
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class CharacterTier(str, Enum):
    """Character tier determines resource allocation and intelligence level"""
    DEDICATED = "dedicated"      # Own Letta agent - full intelligence
    SHARED = "shared"           # Shared agent pool - good intelligence  
    TEMPLATE = "template"       # Static responses - basic intelligence


class UnityCharacterCreate(BaseModel):
    """Request model for creating a new Unity character"""
    
    unity_character_id: str = Field(
        ...,
        description="Unity's internal ID for this character (must be unique)",
        min_length=1,
        max_length=255
    )
    
    character_name: str = Field(
        ...,
        description="Human-readable name of the character",
        min_length=1,
        max_length=255
    )
    
    personality: str = Field(
        ...,
        description="Character personality description for AI agent",
        min_length=10,
        max_length=2000
    )
    
    backstory: str = Field(
        ...,
        description="Character backstory and history",
        min_length=10,
        max_length=2000  
    )
    
    voice_style: Optional[str] = Field(
        None,
        description="Voice/speaking style (e.g., 'cheerful and talkative')",
        max_length=255
    )
    
    role: str = Field(
        ...,
        description="Character role (e.g., 'merchant_npc', 'main_character')",
        min_length=1,
        max_length=100
    )
    
    location: Optional[str] = Field(
        None,
        description="Default location where character appears",
        max_length=255
    )
    
    game_id: str = Field(
        ...,
        description="Identifier for the game/story this character belongs to",
        min_length=1,
        max_length=255
    )
    
    character_tier: CharacterTier = Field(
        CharacterTier.DEDICATED,
        description="Resource tier for this character"
    )
    
    character_config: Optional[Dict] = Field(
        None,
        description="Additional character configuration (animations, behaviors, etc.)"
    )

    @field_validator('unity_character_id')
    @classmethod
    def validate_unity_character_id(cls, v):
        """Ensure Unity character ID follows naming conventions"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Unity character ID cannot be empty")
        
        # Allow alphanumeric, underscores, hyphens
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Unity character ID must contain only alphanumeric characters, underscores, and hyphens")
        
        return v.strip()

    @field_validator('game_id')  
    @classmethod
    def validate_game_id(cls, v):
        """Ensure game ID is properly formatted"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Game ID cannot be empty")
        return v.strip()


class UnityCharacterUpdate(BaseModel):
    """Request model for updating an existing Unity character"""
    
    character_name: Optional[str] = Field(
        None,
        description="Updated character name",
        min_length=1,
        max_length=255
    )
    
    personality: Optional[str] = Field(
        None,
        description="Updated personality description",
        min_length=10,
        max_length=2000
    )
    
    backstory: Optional[str] = Field(
        None,
        description="Updated backstory",
        min_length=10,
        max_length=2000
    )
    
    voice_style: Optional[str] = Field(
        None,
        description="Updated voice style",
        max_length=255
    )
    
    location: Optional[str] = Field(
        None,
        description="Updated default location",
        max_length=255
    )
    
    is_active: Optional[bool] = Field(
        None,
        description="Whether character is active"
    )
    
    character_config: Optional[Dict] = Field(
        None,
        description="Updated character configuration"
    )


class UnityCharacter(BaseModel):
    """Response model for Unity character data"""
    
    id: str = Field(..., description="Internal unique identifier")
    unity_character_id: str = Field(..., description="Unity's character ID")
    letta_agent_id: str = Field(..., description="Letta agent powering this character")
    character_name: str = Field(..., description="Character name")
    personality: str = Field(..., description="Character personality")
    backstory: str = Field(..., description="Character backstory")
    voice_style: Optional[str] = Field(None, description="Voice style")
    role: str = Field(..., description="Character role")
    location: Optional[str] = Field(None, description="Default location")
    game_id: str = Field(..., description="Game identifier")
    character_tier: CharacterTier = Field(..., description="Character tier")
    is_active: bool = Field(..., description="Whether character is active")
    character_config: Optional[Dict] = Field(None, description="Character configuration")
    usage_stats: Optional[Dict] = Field(None, description="Usage statistics")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True


class UnityCharacterRegistrationResponse(BaseModel):
    """Response after successfully registering a Unity character"""
    
    success: bool = Field(..., description="Whether registration succeeded")
    character: UnityCharacter = Field(..., description="The registered character")
    letta_agent_id: str = Field(..., description="ID of created Letta agent")
    dialogue_endpoint: str = Field(..., description="Endpoint for character dialogue")
    instructions: List[str] = Field(..., description="Next steps for Unity team")

    
class UnityCharacterListResponse(BaseModel):
    """Response for listing Unity characters"""
    
    characters: List[UnityCharacter] = Field(..., description="List of characters")
    total: int = Field(..., description="Total number of characters") 
    game_id: Optional[str] = Field(None, description="Game filter applied")


class UnityDialogueRequest(BaseModel):
    """Request for character dialogue generation"""
    
    player_message: str = Field(
        ...,
        description="What the player said to the character",
        min_length=1,
        max_length=1000
    )
    
    scene_context: Optional[Dict] = Field(
        None,
        description="Current scene context (location, time, game state, etc.)"
    )
    
    interaction_type: str = Field(
        "conversation",
        description="Type of interaction (conversation, quest, trade, etc.)"
    )

    @field_validator('player_message')
    @classmethod 
    def validate_player_message(cls, v):
        """Ensure player message is not empty"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Player message cannot be empty")
        return v.strip()


class UnityDialogueResponse(BaseModel):
    """Response from character dialogue generation"""
    
    character_id: str = Field(..., description="Unity character ID")
    dialogue_text: str = Field(..., description="Character's response text")
    emotion: Optional[str] = Field(None, description="Character emotion for animations")
    animation_suggestion: Optional[str] = Field(None, description="Suggested Unity animation")
    memory_updated: bool = Field(..., description="Whether character memory was updated")
    relationship_change: Optional[int] = Field(None, description="Change in relationship score")
    next_action: Optional[str] = Field(None, description="Suggested next action for Unity")


class UnityInstruction(BaseModel):
    """Generic Unity instruction for game state changes"""
    
    instruction_type: str = Field(..., description="Type of instruction (display_dialogue, show_choices, etc.)")
    data: Dict = Field(..., description="Instruction-specific data payload")
    next_action: Optional[str] = Field(None, description="What Unity should do next")
    metadata: Optional[Dict] = Field(None, description="Additional context (timing, effects, etc.)")


class UnityGameStateRequest(BaseModel):
    """Request containing current Unity game state"""
    
    player_id: str = Field(..., description="Player identifier")
    game_id: str = Field(..., description="Game identifier")
    current_scene: Optional[str] = Field(None, description="Current scene/location")
    player_inventory: Optional[List[str]] = Field(None, description="Player inventory items")
    character_relationships: Optional[Dict[str, float]] = Field(None, description="Character relationship scores")
    game_progress: Optional[Dict] = Field(None, description="General game progress data")
    last_action: Optional[str] = Field(None, description="Last action player took")


class UnityCharacterError(BaseModel):
    """Error response for Unity character operations"""
    
    error: bool = Field(True, description="Error flag")
    error_code: str = Field(..., description="Machine-readable error code")
    error_message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict] = Field(None, description="Additional error details")
    suggestions: Optional[List[str]] = Field(None, description="Suggested fixes")
