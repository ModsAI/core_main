"""
Semantic Evaluation Service

Evaluates if dialogue exchanges address story beat objectives using LLM-based semantic analysis.
"""

import re
from typing import Tuple

from letta.llm_api.llm_client import LLMClient
from letta.log import get_logger
from letta.schemas.enums import ProviderType
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message, MessageRole, TextContent
from letta.schemas.user import User

logger = get_logger(__name__)


class SemanticEvaluationService:
    """
    Service for evaluating dialogue beat completion using semantic analysis.
    
    Uses LLM to determine if player dialogue addresses the beat objective.
    Falls back to keyword matching if LLM fails.
    """
    
    def __init__(self):
        """Initialize semantic evaluation service with model configuration."""
        self.model_name = "gemini-2.0-flash-001"
        self.model_endpoint_type = "google_ai"
        self.temperature = 0.1  # Low temp for consistent yes/no
        self.max_retries = 2
        logger.info("🧠 SemanticEvaluationService initialized")
    
    async def evaluate_beat_completion(
        self,
        beat_objective: str,
        player_message: str,
        character_response: str,
        actor: User,
    ) -> Tuple[bool, float]:
        """
        Evaluate if dialogue exchange addressed beat objective.
        
        Args:
            beat_objective: The goal/topic of this beat
            player_message: What the player said
            character_response: How the character responded
            actor: User making the request
            
        Returns:
            (was_addressed: bool, confidence_score: float 0.0-1.0)
        """
        try:
            logger.debug(f"  🧠 Evaluating: '{beat_objective}' vs '{player_message[:50]}...'")
            
            # Build prompt
            prompt_text = self._build_evaluation_prompt(
                beat_objective=beat_objective,
                player_message=player_message,
                character_response=character_response,
            )
            
            # Create LLM config
            llm_config = LLMConfig(
                model=self.model_name,
                model_endpoint_type=self.model_endpoint_type,
                model_endpoint="https://generativelanguage.googleapis.com",
                context_window=1048576,  # Gemini 2.0 Flash context
                temperature=self.temperature,
            )
            
            # Create LLM client
            llm_client = LLMClient.create(
                provider_type=ProviderType.google_ai,
                put_inner_thoughts_first=False,
                actor=actor,
            )
            
            # Build messages
            messages = [
                Message(
                    role=MessageRole.user,
                    content=[TextContent(text=prompt_text)],
                )
            ]
            
            # Make API call
            request_data = llm_client.build_request_data(messages, llm_config, tools=[])
            response_data = await llm_client.request_async(request_data, llm_config)
            response = llm_client.convert_response_to_chat_completion(response_data, messages, llm_config)
            
            # Extract response text
            if response.choices and len(response.choices) > 0:
                response_text = ""
                content = response.choices[0].message.content
                
                # Handle different content formats
                if isinstance(content, str):
                    response_text = content
                elif isinstance(content, list):
                    for item in content:
                        if hasattr(item, 'text'):
                            response_text += item.text
                        elif isinstance(item, dict) and 'text' in item:
                            response_text += item['text']
                
                # Parse response
                was_addressed, confidence_score = self._parse_evaluation_response(response_text)
                logger.debug(f"  ✅ LLM eval: {was_addressed} ({confidence_score:.2f})")
                return (was_addressed, confidence_score)
            else:
                logger.warning("  ⚠️ No response from LLM, using fallback")
                return self._keyword_fallback(beat_objective, player_message, character_response)
                
        except Exception as e:
            logger.warning(f"  ⚠️ Semantic evaluation failed: {e}, falling back to keywords")
            return self._keyword_fallback(beat_objective, player_message, character_response)
    
    def _build_evaluation_prompt(
        self,
        beat_objective: str,
        player_message: str,
        character_response: str,
    ) -> str:
        """
        Build prompt for semantic evaluation.
        
        Returns prompt text for LLM.
        """
        prompt = f"""You are evaluating if a dialogue exchange addresses a story beat objective.

BEAT OBJECTIVE: {beat_objective}

DIALOGUE EXCHANGE:
Player said: "{player_message}"
Character responded: "{character_response}"

TASK: Did this dialogue exchange address the beat objective? Consider:
- Did the player's message relate to the objective?
- Did the character's response engage with the objective?
- Was the objective topic discussed or progressed?

Answer in this EXACT format:
- If YES: "YES (confidence)" where confidence is 0-100
- If NO: "NO (confidence)" where confidence is 0-100

Examples:
- "YES (85)" - Clearly addressed
- "NO (20)" - Completely off-topic
- "YES (60)" - Partially addressed

Your answer:"""
        
        return prompt
    
    def _parse_evaluation_response(self, response_text: str) -> Tuple[bool, float]:
        """
        Parse LLM response to extract YES/NO and confidence.
        
        Expected format: "YES (85)" or "NO (30)"
        
        Returns:
            (was_addressed: bool, confidence_score: float 0.0-1.0)
        """
        try:
            # Extract YES/NO and confidence using regex
            # Pattern: YES or NO followed by (number)
            pattern = r'(YES|NO)\s*\((\d+)\)'
            match = re.search(pattern, response_text.upper())
            
            if match:
                answer = match.group(1)  # YES or NO
                confidence = int(match.group(2))  # 0-100
                
                # Convert to our format
                was_addressed = (answer == "YES")
                confidence_score = confidence / 100.0  # Convert to 0.0-1.0
                
                logger.debug(f"  📊 Parsed: {answer} ({confidence}%) = {was_addressed}, {confidence_score:.2f}")
                return (was_addressed, confidence_score)
            else:
                logger.warning(f"  ⚠️ Could not parse response: {response_text[:100]}")
                return (False, 0.5)  # Default fallback
                
        except Exception as e:
            logger.error(f"  ❌ Error parsing response: {e}")
            return (False, 0.5)
    
    def _keyword_fallback(
        self,
        beat_objective: str,
        player_message: str,
        character_response: str,
    ) -> Tuple[bool, float]:
        """
        Fallback to keyword matching if LLM fails.
        
        Simple keyword extraction and matching.
        Uses the same logic as the existing _beat_was_addressed() method.
        """
        logger.debug(f"  🔑 Using keyword fallback")
        
        # Combine player message and character response
        combined_text = f"{player_message} {character_response}".lower()
        beat_objective_lower = beat_objective.lower()
        
        # Extract keywords from objective (words longer than 3 chars)
        objective_words = [
            word.strip(".,!?:;\"'")
            for word in beat_objective_lower.split()
            if len(word.strip(".,!?:;\"'")) > 3
        ]
        
        # Check how many keywords appear in the dialogue
        matches = sum(1 for keyword in objective_words if keyword in combined_text)
        
        # Calculate match ratio
        if len(objective_words) > 0:
            match_ratio = matches / len(objective_words)
        else:
            match_ratio = 0.0
        
        # Consider it addressed if at least 30% of keywords match
        was_addressed = match_ratio >= 0.3
        confidence_score = 0.5  # Fixed confidence for keyword fallback
        
        logger.debug(f"  🔑 Keyword match: {matches}/{len(objective_words)} = {match_ratio:.2f}, addressed={was_addressed}")
        
        return (was_addressed, confidence_score)

