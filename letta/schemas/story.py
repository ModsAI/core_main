"""
Story System Pydantic Schemas

Data models for story management, session tracking, and dialogue guidance.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class SessionStatus(str, Enum):
    """Status of a story session"""
    ACTIVE = "active"          # Currently being played
    PAUSED = "paused"          # Saved, can be resumed
    COMPLETED = "completed"    # Story finished
    ARCHIVED = "archived"      # Old session, kept for history


class InstructionType(str, Enum):
    """Types of story instructions"""
    SETTING = "setting"        # Scene setup
    NARRATION = "narration"    # Story text
    DIALOGUE = "dialogue"      # Character speech
    ACTION = "action"          # Character action
    END = "end"                # Story end marker


# ============================================================
# Story Definition Models (Input from TypeScript)
# ============================================================

class StoryCharacter(BaseModel):
    """Character definition from story JSON"""
    name: str = Field(..., description="Character name")
    sex: str = Field(..., description="Character gender")
    age: int = Field(..., description="Character age")
    is_main_character: bool = Field(False, alias="isMainCharacter", description="Is this the player character")
    model: Optional[str] = Field(None, description="Unity 3D model name")
    
    # Generated fields
    character_id: Optional[str] = Field(None, description="Auto-generated ID (name lowercase)")
    
    class Config:
        populate_by_name = True  # Allow both 'is_main_character' and 'isMainCharacter'


class StoryInstruction(BaseModel):
    """Instruction from story JSON"""
    type: InstructionType = Field(..., description="Instruction type")
    
    # Setting fields
    title: Optional[str] = Field(None, description="Scene title")
    setting: Optional[str] = Field(None, description="Scene location/environment")
    
    # Text fields
    text: Optional[str] = Field(None, description="Narration or action text")
    
    # Dialogue/Action fields
    character: Optional[str] = Field(None, description="Character name")
    action: Optional[str] = Field(None, description="Action description")


class StoryUpload(BaseModel):
    """Request to upload a new story"""
    id: int = Field(..., description="Story ID")
    title: str = Field(..., description="Story title")
    characters: List[StoryCharacter] = Field(..., description="Story characters")
    instructions: List[StoryInstruction] = Field(..., description="Story instructions")
    
    # Optional metadata
    description: Optional[str] = Field(None, description="Story description")
    tags: Optional[List[str]] = Field(None, description="Story tags")


# ============================================================
# Processed Story Models (Internal Representation)
# ============================================================

class Scene(BaseModel):
    """A scene in the story (group of instructions between settings)"""
    scene_id: str = Field(..., description="Scene identifier")
    scene_number: int = Field(..., description="Scene number in sequence")
    title: str = Field(..., description="Scene title")
    location: str = Field(..., description="Scene location")
    instructions: List[StoryInstruction] = Field(..., description="Instructions in this scene")
    characters: List[str] = Field(..., description="Characters present in scene")
    dialogue_beats: List[Dict[str, Any]] = Field(default_factory=list, description="Dialogue beats to track")


class Story(BaseModel):
    """Complete story with processed scenes"""
    story_id: str = Field(..., description="Story identifier")
    title: str = Field(..., description="Story title")
    description: Optional[str] = Field(None, description="Story description")
    characters: List[StoryCharacter] = Field(..., description="Story characters")
    scenes: List[Scene] = Field(..., description="Story scenes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ============================================================
# Session Management Models
# ============================================================

class SessionCreate(BaseModel):
    """Request to start a new story session"""
    story_id: str = Field(..., description="Story to play")
    player_name: Optional[str] = Field(None, description="Player name (optional)")


class SessionState(BaseModel):
    """Current state of a story session"""
    current_scene_number: int = Field(0, description="Current scene number")
    current_instruction_index: int = Field(0, description="Current instruction index in scene")
    completed_dialogue_beats: List[str] = Field(default_factory=list, description="Completed dialogue beat IDs")
    character_relationships: Dict[str, float] = Field(default_factory=dict, description="Character relationship scores")
    player_choices: List[Dict[str, Any]] = Field(default_factory=list, description="Choices made by player")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Story variables")


class StorySession(BaseModel):
    """A story session (user playing a story)"""
    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User playing the story")
    story_id: str = Field(..., description="Story being played")
    status: SessionStatus = Field(..., description="Session status")
    state: SessionState = Field(..., description="Current session state")
    character_agents: Dict[str, str] = Field(..., description="Character name -> agent ID mapping")
    created_at: datetime = Field(..., description="Session creation time")
    updated_at: datetime = Field(..., description="Last update time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    
    class Config:
        from_attributes = True


class SessionResume(BaseModel):
    """Response when resuming a session"""
    session: StorySession = Field(..., description="Session details")
    current_scene: Scene = Field(..., description="Current scene")
    recent_history: List[Dict[str, Any]] = Field(..., description="Recent interactions")


# ============================================================
# Dialogue Models
# ============================================================

class StoryDialogueRequest(BaseModel):
    """Request for dialogue with script guidance"""
    player_message: str = Field(..., description="What the player said", min_length=1, max_length=2000)
    target_character: str = Field(..., description="Character to talk to")
    
    @field_validator('player_message')
    @classmethod
    def validate_message(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Player message cannot be empty")
        return v.strip()


class DialogueBeatInfo(BaseModel):
    """Information about a dialogue beat"""
    beat_id: str = Field(..., description="Beat identifier")
    character: str = Field(..., description="Character who should say this")
    script_text: Optional[str] = Field(None, description="Scripted text (if exact)")
    topic: str = Field(..., description="What this beat is about")
    is_completed: bool = Field(..., description="Whether this beat has been said")


class StoryDialogueResponse(BaseModel):
    """Response from character with script guidance"""
    character_id: str = Field(..., description="Character who responded")
    character_name: str = Field(..., description="Character name")
    dialogue_text: str = Field(..., description="Character's response")
    emotion: str = Field(..., description="Character emotion")
    animation_suggestion: str = Field(..., description="Suggested animation")
    
    # Script tracking
    dialogue_beats_completed: List[str] = Field(..., description="Beats completed in this response")
    scene_progress: float = Field(..., description="Progress through current scene (0-1)")
    scene_complete: bool = Field(..., description="Is current scene complete")
    
    # Session state
    session_updated: bool = Field(..., description="Was session state updated")
    next_scene_number: Optional[int] = Field(None, description="Next scene number (if transitioning)")


# ============================================================
# Error Models
# ============================================================

class StoryError(BaseModel):
    """Error response for story operations"""
    error: bool = Field(True, description="Error flag")
    error_code: str = Field(..., description="Machine-readable error code")
    error_message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    suggestions: Optional[List[str]] = Field(None, description="Suggested fixes")


# ============================================================
# Response Models
# ============================================================

class StoryUploadResponse(BaseModel):
    """Response after uploading a story"""
    success: bool = Field(..., description="Upload success")
    story_id: str = Field(..., description="Story identifier")
    title: str = Field(..., description="Story title")
    scene_count: int = Field(..., description="Number of scenes")
    character_count: int = Field(..., description="Number of characters")
    instructions: List[str] = Field(..., description="Next steps")


class SessionStartResponse(BaseModel):
    """Response after starting a session"""
    success: bool = Field(..., description="Start success")
    session_id: str = Field(..., description="Session identifier")
    story_title: str = Field(..., description="Story title")
    first_scene: Scene = Field(..., description="First scene")
    current_scene: Scene = Field(..., description="Current scene (alias for first_scene)")
    player_character: Optional[str] = Field(None, description="Player character name")
    available_characters: List[str] = Field(..., description="List of character names available for dialogue")
    instructions: List[str] = Field(..., description="How to proceed")


class SessionRestartResponse(BaseModel):
    """Response after restarting a session"""
    success: bool = Field(..., description="Restart success")
    session_id: str = Field(..., description="New or reset session ID")
    message: str = Field(..., description="What happened")

