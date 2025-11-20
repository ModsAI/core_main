"""
Story Validation Service (IMM-10)

Validates story JSON for common authoring errors before upload.
Provides helpful, actionable error messages to story authors.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Set
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Single validation error with context and suggestions"""
    
    error_type: str  # "missing_character", "invalid_choice_id", etc.
    message: str     # Human-readable error message
    location: str    # Where error occurred (e.g., "instruction 5, choice 2")
    severity: str    # "error" or "warning"
    suggestion: str  # How to fix the error


class StoryValidator:
    """
    Validates story JSON for common authoring errors.
    
    Checks:
    - Character definitions and references
    - Choice IDs and references
    - Relationship definitions and references
    - Conditional logic validity
    - Story structure requirements
    - Instruction type requirements
    """
    
    def validate_story(self, story_data: Dict[str, Any]) -> List[ValidationError]:
        """
        Comprehensive story validation.
        
        Args:
            story_data: Story JSON as dictionary
            
        Returns:
            List of ValidationError objects (empty if valid)
        """
        errors = []
        
        try:
            logger.info("Starting story validation...")
            
            # Collect data for cross-validation
            self.character_names: Set[str] = set()
            self.all_choice_ids: Set[int] = set()
            self.all_relationship_ids: Set[str] = set()
            
            # Run all validation checks
            logger.info("Validating basic structure...")
            errors.extend(self._validate_basic_structure(story_data))
            
            logger.info("Validating characters...")
            errors.extend(self._validate_characters(story_data))
            
            logger.info("Validating relationships...")
            errors.extend(self._validate_relationships(story_data))
            
            logger.info("Validating instructions...")
            errors.extend(self._validate_instructions(story_data))
            
            logger.info("Validating choices...")
            errors.extend(self._validate_choices(story_data))
            
            logger.info("Validating conditionals...")
            errors.extend(self._validate_conditionals(story_data))
            
            logger.info(f"Validation complete. Found {len(errors)} errors/warnings")
            
        except Exception as e:
            logger.error(f"CRITICAL: Validator crashed with exception: {type(e).__name__}: {e}")
            logger.exception("Full traceback:")
            # Create a critical error to return instead of crashing
            errors.append(ValidationError(
                error_type="validator_crash",
                message=f"Internal validator error: {type(e).__name__}: {str(e)}",
                location="validator",
                severity="error",
                suggestion="This is a bug in the validator. Please report this error."
            ))
        
        return errors
    
    def _validate_basic_structure(self, story_data: Dict) -> List[ValidationError]:
        """Validate basic story structure"""
        errors = []
        
        try:
            # Check required top-level fields
            if not story_data.get("id"):
                errors.append(ValidationError(
                    error_type="missing_story_id",
                    message="Story must have an 'id' field",
                    location="root",
                    severity="error",
                    suggestion="Add 'id' field with a unique integer (e.g., 12345)"
                ))
            
            if not story_data.get("title"):
                errors.append(ValidationError(
                    error_type="missing_story_title",
                    message="Story must have a 'title' field",
                    location="root",
                    severity="error",
                    suggestion="Add 'title' field with story name"
                ))
            
            # Check instructions exist
            instructions = story_data.get("instructions", [])
            if not instructions:
                errors.append(ValidationError(
                    error_type="no_instructions",
                    message="Story must have at least one instruction",
                    location="root",
                    severity="error",
                    suggestion="Add 'instructions' array with story content"
                ))
        
        except Exception as e:
            logger.error(f"Error in _validate_basic_structure: {e}")
            errors.append(ValidationError(
                error_type="validation_error",
                message=f"Error validating basic structure: {str(e)}",
                location="basic_structure",
                severity="error",
                suggestion="Check story JSON format"
            ))
        
        return errors
    
    def _validate_characters(self, story_data: Dict) -> List[ValidationError]:
        """Validate character definitions"""
        errors = []
        
        try:
            characters = story_data.get("characters", [])
            logger.info(f"  Validating {len(characters)} characters...")
            
            if not characters:
                errors.append(ValidationError(
                    error_type="no_characters",
                    message="Story must have at least one character",
                    location="root",
                    severity="error",
                    suggestion="Add 'characters' array with at least one character"
                ))
                return errors  # Can't continue without characters
            
            # Check for main character (support both camelCase and snake_case)
            main_chars = [c for c in characters if c.get("isMainCharacter") or c.get("is_main_character")]
            logger.info(f"  Found {len(main_chars)} main character(s)")
            
            if len(main_chars) == 0:
                errors.append(ValidationError(
                    error_type="no_main_character",
                    message="Story must have exactly one main character",
                    location="characters",
                    severity="error",
                    suggestion="Set 'isMainCharacter: true' for one character"
                ))
            elif len(main_chars) > 1:
                errors.append(ValidationError(
                    error_type="multiple_main_characters",
                    message=f"Story has {len(main_chars)} main characters (must have exactly 1)",
                    location="characters",
                    severity="error",
                    suggestion="Set 'isMainCharacter: true' for only ONE character"
                ))
            
            # Check for duplicate names
            names = [c.get("name") for c in characters if c.get("name")]
            duplicates = {name for name in names if names.count(name) > 1}
            if duplicates:
                errors.append(ValidationError(
                    error_type="duplicate_character_names",
                    message=f"Duplicate character names: {', '.join(duplicates)}",
                    location="characters",
                    severity="error",
                    suggestion="Each character must have a unique name"
                ))
            
            # Store character names for reference validation
            self.character_names = set(names)
            logger.info(f"  Character names: {self.character_names}")
            
            # Check required character fields
            for idx, char in enumerate(characters):
                if not char.get("name"):
                    errors.append(ValidationError(
                        error_type="missing_character_name",
                        message=f"Character {idx + 1} is missing 'name' field",
                        location=f"characters[{idx}]",
                        severity="error",
                        suggestion="Add 'name' field to character"
                    ))
        
        except Exception as e:
            logger.error(f"Error in _validate_characters: {e}")
            logger.exception("Full traceback:")
            errors.append(ValidationError(
                error_type="validation_error",
                message=f"Error validating characters: {str(e)}",
                location="characters",
                severity="error",
                suggestion="Check characters array format"
            ))
            
            if not char.get("age"):
                errors.append(ValidationError(
                    error_type="missing_character_age",
                    message=f"Character '{char.get('name', idx + 1)}' is missing 'age' field",
                    location=f"characters[{idx}]",
                    severity="error",
                    suggestion="Add 'age' field to character"
                ))
            
            if not char.get("sex"):
                errors.append(ValidationError(
                    error_type="missing_character_sex",
                    message=f"Character '{char.get('name', idx + 1)}' is missing 'sex' field",
                    location=f"characters[{idx}]",
                    severity="error",
                    suggestion="Add 'sex' field to character (e.g., 'male', 'female', 'neutral')"
                ))
        
        return errors
    
    def _validate_relationships(self, story_data: Dict) -> List[ValidationError]:
        """Validate relationship definitions"""
        errors = []
        relationships = story_data.get("relationships", [])
        
        if not relationships:
            # This is optional, so just warn
            return errors
        
        for idx, rel in enumerate(relationships):
            char_name = rel.get("character")
            rel_type = rel.get("type")
            
            # Check character exists
            if char_name and char_name not in self.character_names:
                errors.append(ValidationError(
                    error_type="invalid_relationship_character",
                    message=f"Relationship {idx + 1} references unknown character '{char_name}'",
                    location=f"relationships[{idx}].character",
                    severity="error",
                    suggestion=f"Character '{char_name}' not in characters array. Add it or fix the name."
                ))
            
            # Check required fields (support both camelCase and snake_case)
            # Use 'is not None' checks to properly handle 0 values
            points_per_level = rel.get("pointsPerLevel") if rel.get("pointsPerLevel") is not None else rel.get("points_per_level")
            if points_per_level is None:
                errors.append(ValidationError(
                    error_type="missing_points_per_level",
                    message=f"Relationship {idx + 1} ({char_name}-{rel_type}) missing 'pointsPerLevel'",
                    location=f"relationships[{idx}]",
                    severity="error",
                    suggestion="Add 'pointsPerLevel' field (e.g., 50)"
                ))
            elif points_per_level <= 0:
                errors.append(ValidationError(
                    error_type="invalid_points_per_level",
                    message=f"Relationship {idx + 1}: 'pointsPerLevel' must be positive",
                    location=f"relationships[{idx}].pointsPerLevel",
                    severity="error",
                    suggestion=f"Change pointsPerLevel to a positive number (currently {points_per_level})"
                ))
            
            max_levels = rel.get("maxLevels") if rel.get("maxLevels") is not None else rel.get("max_levels")
            if max_levels is None:
                errors.append(ValidationError(
                    error_type="missing_max_levels",
                    message=f"Relationship {idx + 1} ({char_name}-{rel_type}) missing 'maxLevels'",
                    location=f"relationships[{idx}]",
                    severity="error",
                    suggestion="Add 'maxLevels' field (e.g., 5)"
                ))
            elif max_levels <= 0:
                errors.append(ValidationError(
                    error_type="invalid_max_levels",
                    message=f"Relationship {idx + 1}: 'maxLevels' must be positive",
                    location=f"relationships[{idx}].maxLevels",
                    severity="error",
                    suggestion=f"Change maxLevels to a positive number (currently {max_levels})"
                ))
            
            starting_points = rel.get("startingPoints") if rel.get("startingPoints") is not None else rel.get("starting_points")
            if starting_points is None:
                errors.append(ValidationError(
                    error_type="missing_starting_points",
                    message=f"Relationship {idx + 1} ({char_name}-{rel_type}) missing 'startingPoints'",
                    location=f"relationships[{idx}]",
                    severity="error",
                    suggestion="Add 'startingPoints' field (e.g., 0)"
                ))
            elif starting_points < 0:
                errors.append(ValidationError(
                    error_type="invalid_starting_points",
                    message=f"Relationship {idx + 1}: 'startingPoints' cannot be negative",
                    location=f"relationships[{idx}].startingPoints",
                    severity="error",
                    suggestion=f"Change startingPoints to 0 or positive number (currently {starting_points})"
                ))
            
            # Check visual fields
            visual = rel.get("visual", {})
            if not visual.get("color"):
                errors.append(ValidationError(
                    error_type="missing_visual_color",
                    message=f"Relationship {idx + 1} missing 'visual.color'",
                    location=f"relationships[{idx}].visual",
                    severity="error",
                    suggestion="Add 'visual.color' field (e.g., '#3498db')"
                ))
            
            if not visual.get("icon"):
                errors.append(ValidationError(
                    error_type="missing_visual_icon",
                    message=f"Relationship {idx + 1} missing 'visual.icon'",
                    location=f"relationships[{idx}].visual",
                    severity="error",
                    suggestion="Add 'visual.icon' field (e.g., 'heart', 'star')"
                ))
            
            # Store relationship ID for reference validation
            if char_name and rel_type:
                # Generate expected relationship ID
                char_id = char_name.lower()
                rel_id = f"{char_id}-{rel_type}"
                self.all_relationship_ids.add(rel_id)
        
        return errors
    
    def _validate_instructions(self, story_data: Dict) -> List[ValidationError]:
        """Validate instruction definitions"""
        errors = []
        
        try:
            instructions = story_data.get("instructions", [])
            logger.info(f"  Validating {len(instructions)} instructions...")
            
            for idx, instruction in enumerate(instructions):
                inst_type = instruction.get("type")
                
                # Check type is valid
                valid_types = ["setting", "narration", "dialogue", "action", "end"]
                if not inst_type:
                    errors.append(ValidationError(
                        error_type="missing_instruction_type",
                        message=f"Instruction {idx + 1} missing 'type' field",
                        location=f"instructions[{idx}]",
                        severity="error",
                        suggestion=f"Add 'type' field (one of: {', '.join(valid_types)})"
                    ))
                elif inst_type not in valid_types:
                    errors.append(ValidationError(
                        error_type="invalid_instruction_type",
                        message=f"Instruction {idx + 1} has invalid type '{inst_type}'",
                        location=f"instructions[{idx}].type",
                        severity="error",
                        suggestion=f"Use one of: {', '.join(valid_types)}"
                    ))
                
                # Type-specific validation
                if inst_type == "dialogue":
                    char_name = instruction.get("character")
                    if not char_name:
                        errors.append(ValidationError(
                            error_type="missing_dialogue_character",
                            message=f"Instruction {idx + 1}: Dialogue instruction missing 'character' field",
                            location=f"instructions[{idx}]",
                            severity="error",
                            suggestion="Add 'character' field with character name"
                        ))
                    elif char_name not in self.character_names:
                        errors.append(ValidationError(
                            error_type="invalid_character_reference",
                            message=f"Instruction {idx + 1} references unknown character '{char_name}'",
                            location=f"instructions[{idx}].character",
                            severity="error",
                            suggestion=f"Add character '{char_name}' to characters array or fix the name"
                        ))
                
                elif inst_type == "action":
                    char_name = instruction.get("character")
                    if char_name and char_name not in self.character_names:
                        errors.append(ValidationError(
                            error_type="invalid_character_reference",
                            message=f"Instruction {idx + 1} references unknown character '{char_name}'",
                            location=f"instructions[{idx}].character",
                            severity="error",
                            suggestion=f"Add character '{char_name}' to characters array or fix the name"
                        ))
                
                elif inst_type in ["setting", "narration"]:
                    if not instruction.get("text"):
                        errors.append(ValidationError(
                            error_type="missing_instruction_text",
                            message=f"Instruction {idx + 1}: {inst_type} instruction missing 'text' field",
                            location=f"instructions[{idx}]",
                            severity="error",
                            suggestion="Add 'text' field with instruction content"
                        ))
        
        except Exception as e:
            logger.error(f"Error in _validate_instructions: {e}")
            logger.exception("Full traceback:")
            errors.append(ValidationError(
                error_type="validation_error",
                message=f"Error validating instructions: {str(e)}",
                location="instructions",
                severity="error",
                suggestion="Check instructions array format"
            ))
        
        return errors
    
    def _validate_choices(self, story_data: Dict) -> List[ValidationError]:
        """Validate choice definitions"""
        errors = []
        instructions = story_data.get("instructions", [])
        
        for idx, instruction in enumerate(instructions):
            choices = instruction.get("choices", [])
            if not choices:
                continue
            
            # Check choices only on valid instruction types
            inst_type = instruction.get("type")
            if inst_type not in ["narration", "dialogue"]:
                errors.append(ValidationError(
                    error_type="invalid_choice_location",
                    message=f"Instruction {idx + 1}: Choices can only be on narration or dialogue instructions",
                    location=f"instructions[{idx}].choices",
                    severity="error",
                    suggestion=f"Move choices to a narration or dialogue instruction (currently on {inst_type})"
                ))
            
            instruction_choice_ids = set()
            
            for choice_idx, choice in enumerate(choices):
                choice_id = choice.get("id")
                
                # Check ID exists
                if choice_id is None:
                    errors.append(ValidationError(
                        error_type="missing_choice_id",
                        message=f"Instruction {idx + 1}, choice {choice_idx + 1}: Missing 'id' field",
                        location=f"instructions[{idx}].choices[{choice_idx}]",
                        severity="error",
                        suggestion=f"Add 'id' field with positive integer (e.g., {choice_idx + 1})"
                    ))
                    continue
                
                # Check ID is positive integer
                if not isinstance(choice_id, int) or choice_id <= 0:
                    errors.append(ValidationError(
                        error_type="invalid_choice_id",
                        message=f"Instruction {idx + 1}, choice {choice_idx + 1}: ID must be positive integer",
                        location=f"instructions[{idx}].choices[{choice_idx}].id",
                        severity="error",
                        suggestion=f"Change choice ID to a positive integer (e.g., {choice_idx + 1})"
                    ))
                
                # Check unique within instruction
                if choice_id in instruction_choice_ids:
                    errors.append(ValidationError(
                        error_type="duplicate_choice_id",
                        message=f"Instruction {idx + 1}: Duplicate choice ID {choice_id}",
                        location=f"instructions[{idx}].choices",
                        severity="error",
                        suggestion="Each choice in an instruction must have a unique ID"
                    ))
                else:
                    instruction_choice_ids.add(choice_id)
                    self.all_choice_ids.add(choice_id)
                
                # Check text exists
                if not choice.get("text"):
                    errors.append(ValidationError(
                        error_type="missing_choice_text",
                        message=f"Instruction {idx + 1}, choice {choice_idx + 1}: Missing 'text' field",
                        location=f"instructions[{idx}].choices[{choice_idx}]",
                        severity="error",
                        suggestion="Add 'text' field with choice description"
                    ))
                
                # Validate relationship effects (support both camelCase and snake_case)
                rel_effects = choice.get("relationshipEffects") or choice.get("relationship_effects") or []
                for effect_idx, effect in enumerate(rel_effects):
                    char_name = effect.get("character")
                    rel_type = effect.get("type")
                    
                    # Check character exists
                    if char_name and char_name not in self.character_names:
                        errors.append(ValidationError(
                            error_type="invalid_effect_character",
                            message=f"Instruction {idx + 1}, choice {choice_idx + 1}: Effect references unknown character '{char_name}'",
                            location=f"instructions[{idx}].choices[{choice_idx}].relationshipEffects[{effect_idx}]",
                            severity="error",
                            suggestion=f"Add character '{char_name}' to characters array or fix the name"
                        ))
                    
                    # Check effect field exists
                    if not effect.get("effect"):
                        errors.append(ValidationError(
                            error_type="missing_effect_value",
                            message=f"Instruction {idx + 1}, choice {choice_idx + 1}: Missing 'effect' field",
                            location=f"instructions[{idx}].choices[{choice_idx}].relationshipEffects[{effect_idx}]",
                            severity="error",
                            suggestion="Add 'effect' field (e.g., '+10', '-5')"
                        ))
        
        return errors
    
    def _validate_conditionals(self, story_data: Dict) -> List[ValidationError]:
        """Validate conditional logic"""
        errors = []
        instructions = story_data.get("instructions", [])
        
        for idx, instruction in enumerate(instructions):
            conditional = instruction.get("conditional")
            if not conditional:
                continue
            
            req_type = conditional.get("requirement_type")
            
            # Check requirement_type exists
            if not req_type:
                errors.append(ValidationError(
                    error_type="missing_conditional_type",
                    message=f"Instruction {idx + 1}: Conditional missing 'requirement_type'",
                    location=f"instructions[{idx}].conditional",
                    severity="error",
                    suggestion="Add 'requirement_type' (e.g., 'choice_made', 'relationship_level')"
                ))
                continue
            
            # Validate based on type
            if req_type in ["choice_made", "player_choice"]:  # Support both for compatibility
                choice_id = conditional.get("choice_id")
                if choice_id is None:
                    errors.append(ValidationError(
                        error_type="missing_conditional_choice_id",
                        message=f"Instruction {idx + 1}: 'choice_made' conditional missing 'choice_id'",
                        location=f"instructions[{idx}].conditional",
                        severity="error",
                        suggestion="Add 'choice_id' field with valid choice ID"
                    ))
                elif choice_id not in self.all_choice_ids:
                    valid_ids = sorted(list(self.all_choice_ids)) if self.all_choice_ids else []
                    errors.append(ValidationError(
                        error_type="invalid_conditional_choice",
                        message=f"Instruction {idx + 1}: Conditional references non-existent choice ID {choice_id}",
                        location=f"instructions[{idx}].conditional.choice_id",
                        severity="error",
                        suggestion=f"Use one of these valid choice IDs: {valid_ids}" if valid_ids else "No choices defined in story yet"
                    ))
            
            elif req_type == "relationship_level":
                rel_id = conditional.get("relationship_id")
                if not rel_id:
                    errors.append(ValidationError(
                        error_type="missing_conditional_relationship_id",
                        message=f"Instruction {idx + 1}: 'relationship_level' conditional missing 'relationship_id'",
                        location=f"instructions[{idx}].conditional",
                        severity="error",
                        suggestion="Add 'relationship_id' field (e.g., 'emma-romance')"
                    ))
                elif rel_id not in self.all_relationship_ids:
                    valid_ids = sorted(list(self.all_relationship_ids)) if self.all_relationship_ids else []
                    errors.append(ValidationError(
                        error_type="invalid_conditional_relationship",
                        message=f"Instruction {idx + 1}: Conditional references non-existent relationship '{rel_id}'",
                        location=f"instructions[{idx}].conditional.relationship_id",
                        severity="error",
                        suggestion=f"Use one of these valid relationship IDs: {valid_ids}" if valid_ids else "No relationships defined in story"
                    ))
                
                # Check min_level or max_level exists
                if conditional.get("min_level") is None and conditional.get("max_level") is None:
                    errors.append(ValidationError(
                        error_type="missing_conditional_level_requirement",
                        message=f"Instruction {idx + 1}: 'relationship_level' conditional needs 'min_level' or 'max_level'",
                        location=f"instructions[{idx}].conditional",
                        severity="error",
                        suggestion="Add 'min_level' (e.g., 3) or 'max_level' (e.g., 2)"
                    ))
            
            elif req_type in ["all", "any"]:
                sub_conditions = conditional.get("sub_conditions")
                if not sub_conditions:
                    errors.append(ValidationError(
                        error_type="missing_sub_conditions",
                        message=f"Instruction {idx + 1}: '{req_type}' conditional requires 'sub_conditions' array",
                        location=f"instructions[{idx}].conditional",
                        severity="error",
                        suggestion="Add 'sub_conditions' array with at least one condition"
                    ))
                elif not isinstance(sub_conditions, list) or len(sub_conditions) == 0:
                    errors.append(ValidationError(
                        error_type="empty_sub_conditions",
                        message=f"Instruction {idx + 1}: 'sub_conditions' must have at least one condition",
                        location=f"instructions[{idx}].conditional.sub_conditions",
                        severity="error",
                        suggestion="Add at least one sub-condition to the array"
                    ))
        
        return errors

