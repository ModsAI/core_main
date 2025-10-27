import json
import uuid
from typing import List, Optional

from google import genai
from google.genai.types import (
    FunctionCallingConfig,
    FunctionCallingConfigMode,
    GenerateContentResponse,
    HttpOptions,
    ThinkingConfig,
    ToolConfig,
)

from letta.constants import NON_USER_MSG_PREFIX
from letta.helpers.datetime_helpers import get_utc_time_int
from letta.helpers.json_helpers import json_dumps, json_loads
from letta.llm_api.llm_client_base import LLMClientBase
from letta.local_llm.json_parser import clean_json_string_extra_backslash
from letta.local_llm.utils import count_tokens
from letta.log import get_logger
from letta.otel.tracing import trace_method
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message as PydanticMessage
from letta.schemas.openai.chat_completion_request import Tool
from letta.schemas.openai.chat_completion_response import ChatCompletionResponse, Choice, FunctionCall, Message, ToolCall, UsageStatistics
from letta.settings import model_settings, settings
from letta.utils import get_tool_call_id

logger = get_logger(__name__)


class GoogleVertexClient(LLMClientBase):

    def _get_client(self):
        timeout_ms = int(settings.llm_request_timeout_seconds * 1000)
        return genai.Client(
            vertexai=True,
            project=model_settings.google_cloud_project,
            location=model_settings.google_cloud_location,
            http_options=HttpOptions(api_version="v1", timeout=timeout_ms),
        )

    @trace_method
    def request(self, request_data: dict, llm_config: LLMConfig) -> dict:
        """
        Performs underlying request to llm and returns raw response.
        """
        client = self._get_client()
        response = client.models.generate_content(
            model=llm_config.model,
            contents=request_data["contents"],
            config=request_data["config"],
        )
        return response.model_dump()

    @trace_method
    async def request_async(self, request_data: dict, llm_config: LLMConfig) -> dict:
        """
        Performs underlying request to llm and returns raw response.
        Includes automatic retry logic for transient errors.
        """
        import asyncio
        from letta.errors import ContextWindowExceededError, LLMBadRequestError, LLMRateLimitError, LLMServerError
        
        client = self._get_client()
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = await client.aio.models.generate_content(
                    model=llm_config.model,
                    contents=request_data["contents"],
                    config=request_data["config"],
                )
                return response.model_dump()
                
            except Exception as e:
                error_str = str(e).lower()
                is_last_attempt = (attempt == max_retries - 1)
                
                # Determine if error is retryable
                is_retryable = any([
                    "timeout" in error_str,
                    "deadline" in error_str,
                    "503" in error_str,
                    "502" in error_str,
                    "500" in error_str and "transient" in error_str,
                    "unavailable" in error_str,
                ])
                
                # Don't retry these errors
                is_non_retryable = any([
                    "authentication" in error_str,
                    "api key" in error_str,
                    "unauthorized" in error_str,
                    "context length" in error_str,
                    "too large" in error_str,
                    "quota" in error_str,
                    "invalid" in error_str and "schema" in error_str,
                ])
                
                if is_non_retryable or is_last_attempt:
                    # Don't retry, re-raise with proper error type
                    raise e
                
                if is_retryable:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"[Google AI] Retryable error on attempt {attempt + 1}/{max_retries}: {str(e)[:200]}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Unknown error type, don't retry
                    raise e
        
        # Should not reach here, but just in case
        raise Exception("Max retries exceeded")

    @staticmethod
    def add_dummy_model_messages(messages: List[dict]) -> List[dict]:
        """Google AI API requires all function call returns are immediately followed by a 'model' role message.

        In Letta, the 'model' will often call a function (e.g. send_message) that itself yields to the user,
        so there is no natural follow-up 'model' role message.

        To satisfy the Google AI API restrictions, we can add a dummy 'yield' message
        with role == 'model' that is placed in-betweeen and function output
        (role == 'tool') and user message (role == 'user').
        """
        dummy_yield_message = {
            "role": "model",
            "parts": [{"text": f"{NON_USER_MSG_PREFIX}Function call returned, waiting for user response."}],
        }
        messages_with_padding = []
        for i, message in enumerate(messages):
            messages_with_padding.append(message)
            # Check if the current message role is 'tool' and the next message role is 'user'
            if message["role"] in ["tool", "function"] and (i + 1 < len(messages) and messages[i + 1]["role"] == "user"):
                messages_with_padding.append(dummy_yield_message)

        return messages_with_padding

    def _clean_google_ai_schema_properties(self, schema_part: dict):
        """Recursively clean schema parts to remove unsupported Google AI keywords."""
        if not isinstance(schema_part, dict):
            return

        # Per https://ai.google.dev/gemini-api/docs/function-calling?example=meeting#notes_and_limitations
        # * Only a subset of the OpenAPI schema is supported.
        # * Supported parameter types in Python are limited.
        unsupported_keys = ["default", "exclusiveMaximum", "exclusiveMinimum", "additionalProperties", "$schema"]
        keys_to_remove_at_this_level = [key for key in unsupported_keys if key in schema_part]
        for key_to_remove in keys_to_remove_at_this_level:
            logger.warning(f"Removing unsupported keyword 	'{key_to_remove}' from schema part.")
            del schema_part[key_to_remove]

        if schema_part.get("type") == "string" and "format" in schema_part:
            allowed_formats = ["enum", "date-time"]
            if schema_part["format"] not in allowed_formats:
                logger.warning(f"Removing unsupported format 	'{schema_part['format']}' for string type. Allowed: {allowed_formats}")
                del schema_part["format"]

        # Check properties within the current level
        if "properties" in schema_part and isinstance(schema_part["properties"], dict):
            for prop_name, prop_schema in schema_part["properties"].items():
                self._clean_google_ai_schema_properties(prop_schema)

        # Check items within arrays
        if "items" in schema_part and isinstance(schema_part["items"], dict):
            self._clean_google_ai_schema_properties(schema_part["items"])

        # Check within anyOf, allOf, oneOf lists
        for key in ["anyOf", "allOf", "oneOf"]:
            if key in schema_part and isinstance(schema_part[key], list):
                for item_schema in schema_part[key]:
                    self._clean_google_ai_schema_properties(item_schema)

    def convert_tools_to_google_ai_format(self, tools: List[Tool], llm_config: LLMConfig) -> List[dict]:
        """
        OpenAI style:
        "tools": [{
            "type": "function",
            "function": {
                "name": "find_movies",
                "description": "find ....",
                "parameters": {
                "type": "object",
                "properties": {
                    PARAM: {
                    "type": PARAM_TYPE,  # eg "string"
                    "description": PARAM_DESCRIPTION,
                    },
                    ...
                },
                "required": List[str],
                }
            }
        }
        ]

        Google AI style:
        "tools": [{
            "functionDeclarations": [{
            "name": "find_movies",
            "description": "find movie titles currently playing in theaters based on any description, genre, title words, etc.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                "location": {
                    "type": "STRING",
                    "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
                },
                "description": {
                    "type": "STRING",
                    "description": "Any kind of description including category or genre, title words, attributes, etc."
                }
                },
                "required": ["description"]
            }
            }, {
            "name": "find_theaters",
            ...
        """
        function_list = [
            dict(
                name=t.function.name,
                description=t.function.description,
                parameters=t.function.parameters,  # TODO need to unpack
            )
            for t in tools
        ]

        # Add inner thoughts if needed
        for func in function_list:
            # Note: Google AI API used to have weird casing requirements, but not any more

            # Google AI API only supports a subset of OpenAPI 3.0, so unsupported params must be cleaned
            if "parameters" in func and isinstance(func["parameters"], dict):
                self._clean_google_ai_schema_properties(func["parameters"])

            # Add inner thoughts
            if llm_config.put_inner_thoughts_in_kwargs:
                from letta.local_llm.constants import INNER_THOUGHTS_KWARG_DESCRIPTION, INNER_THOUGHTS_KWARG_VERTEX

                func["parameters"]["properties"][INNER_THOUGHTS_KWARG_VERTEX] = {
                    "type": "string",
                    "description": INNER_THOUGHTS_KWARG_DESCRIPTION,
                }
                func["parameters"]["required"].append(INNER_THOUGHTS_KWARG_VERTEX)

        return [{"functionDeclarations": function_list}]

    @trace_method
    def build_request_data(
        self,
        messages: List[PydanticMessage],
        llm_config: LLMConfig,
        tools: List[dict],
        force_tool_call: Optional[str] = None,
    ) -> dict:
        """
        Constructs a request object in the expected data format for this client.
        Includes validation to catch issues before sending to Gemini.
        """
        from letta.errors import LLMBadRequestError, ErrorCode

        # Validate tool count - Gemini has limits on function declarations
        if tools and len(tools) > 64:
            logger.warning(f"[Google AI] Tool count ({len(tools)}) exceeds recommended limit of 64. This may cause request failures.")
            # Don't fail hard, but warn - some models may support more

        if tools:
            tool_objs = [Tool(type="function", function=t) for t in tools]
            tool_names = [t.function.name for t in tool_objs]
            
            # Validate tool names
            for tool_name in tool_names:
                if not tool_name or len(tool_name) > 64:
                    raise LLMBadRequestError(
                        message=f"Tool name '{tool_name}' is invalid (must be 1-64 characters)",
                        code=ErrorCode.INVALID_ARGUMENT,
                        details={"tool_name": tool_name}
                    )
            
            # Convert to the exact payload style Google expects
            try:
                formatted_tools = self.convert_tools_to_google_ai_format(tool_objs, llm_config)
            except Exception as e:
                logger.error(f"[Google AI] Failed to convert tools to Google AI format: {e}")
                raise LLMBadRequestError(
                    message=f"Failed to format tools for Google AI. Tool schemas may be too complex or contain unsupported features: {e}",
                    code=ErrorCode.INVALID_ARGUMENT,
                    details={"error": str(e), "tool_count": len(tools)}
                )
        else:
            formatted_tools = []
            tool_names = []

        # Validate message count
        if len(messages) > 1000:
            logger.warning(f"[Google AI] Message count ({len(messages)}) is very high. Consider summarizing conversation history.")

        try:
            contents = self.add_dummy_model_messages(
                [m.to_google_ai_dict() for m in messages],
            )
        except Exception as e:
            logger.error(f"[Google AI] Failed to convert messages to Google AI format: {e}")
            raise LLMBadRequestError(
                message=f"Failed to format messages for Google AI: {e}",
                code=ErrorCode.INVALID_ARGUMENT,
                details={"error": str(e), "message_count": len(messages)}
            )

        request_data = {
            "contents": contents,
            "config": {
                "temperature": llm_config.temperature,
                "max_output_tokens": llm_config.max_tokens,
                "tools": formatted_tools,
            },
        }

        if len(tool_names) == 1 and settings.use_vertex_structured_outputs_experimental:
            request_data["config"]["response_mime_type"] = "application/json"
            request_data["config"]["response_schema"] = self.get_function_call_response_schema(tools[0])
            del request_data["config"]["tools"]
        else:
            # Note: allowed_function_names can ONLY be set when mode is ANY, not AUTO
            # Gemini API rejects requests with allowed_function_names when mode is AUTO
            tool_config = ToolConfig(
                function_calling_config=FunctionCallingConfig(
                    # AUTO mode lets the model choose between text and function calls intelligently
                    # This should make the model more likely to use memory tools when appropriate
                    mode=FunctionCallingConfigMode.AUTO,
                    # Don't set allowed_function_names with AUTO mode - API will reject it
                )
            )
            request_data["config"]["tool_config"] = tool_config.model_dump()

        # Add thinking_config for flash
        # If enable_reasoner is False, set thinking_budget to 0
        # Otherwise, use the value from max_reasoning_tokens
        if "flash" in llm_config.model:
            thinking_config = ThinkingConfig(
                thinking_budget=llm_config.max_reasoning_tokens if llm_config.enable_reasoner else 0,
            )
            request_data["config"]["thinking_config"] = thinking_config.model_dump()

        # Estimate request size and warn if potentially too large
        try:
            request_json = json.dumps(request_data)
            request_size_mb = len(request_json) / (1024 * 1024)
            if request_size_mb > 10:
                logger.warning(f"[Google AI] Request size is large ({request_size_mb:.2f} MB). This may cause failures or slow responses.")
        except Exception:
            pass  # Don't fail on size estimation

        return request_data

    @trace_method
    def convert_response_to_chat_completion(
        self,
        response_data: dict,
        input_messages: List[PydanticMessage],
        llm_config: LLMConfig,
    ) -> ChatCompletionResponse:
        """
        Converts custom response format from llm client into an OpenAI
        ChatCompletionsResponse object.

        Example:
        {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": " OK. Barbie is showing in two theaters in Mountain View, CA: AMC Mountain View 16 and Regal Edwards 14."
                        }
                    ]
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 9,
            "candidatesTokenCount": 27,
            "totalTokenCount": 36
        }
        }
        """
        # print(response_data)

        response = GenerateContentResponse(**response_data)
        try:
            choices = []
            index = 0
            for candidate in response.candidates:
                content = candidate.content

                if content.role is None or content.parts is None:
                    # This means the response is malformed like MALFORMED_FUNCTION_CALL
                    # 🚨 MALFORMED_FUNCTION_CALL FIX: Handle gracefully for image generation
                    if candidate.finish_reason == "MALFORMED_FUNCTION_CALL":
                        from letta.log import get_logger
                        logger = get_logger(__name__)
                        logger.warning(f"🚨 MALFORMED_FUNCTION_CALL detected: {candidate.finish_message[:350]}...")
                        
                        # Check if this is an image generation request
                        if hasattr(candidate, 'finish_message') and candidate.finish_message:
                            message = candidate.finish_message.lower()
                            if any(keyword in message for keyword in ['image', 'generate', 'picture', 'photo']):
                                logger.info("🎨 Detected malformed image generation call - providing fallback")
                                # Create a fallback function call for image generation
                                import json
                                import uuid
                                
                                # Extract prompt from the malformed message if possible
                                prompt = "beautiful image"  # default
                                try:
                                    # Try to extract prompt from the malformed message
                                    if "prompt" in message:
                                        prompt_start = message.find("prompt") + 6
                                        prompt_part = message[prompt_start:prompt_start+100]
                                        if prompt_part.strip():
                                            prompt = prompt_part.strip()[:50]  # Limit length
                                except:
                                    pass
                                
                                # Create a proper function call structure
                                fallback_function_call = {
                                    "name": "generate_image",
                                    "arguments": json.dumps({"prompt": prompt})
                                }
                                
                                # Create a mock content structure
                                from openai.types.chat.chat_completion_message_tool_call import Function as OpenAIFunction
                                from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall as OpenAIToolCall
                                
                                mock_tool_call = OpenAIToolCall(
                                    id=f"call_{uuid.uuid4().hex[:8]}",
                                    type="function",
                                    function=OpenAIFunction(
                                        name="generate_image",
                                        arguments=json.dumps({"prompt": prompt})
                                    )
                                )
                                
                                # Create a choice with the fallback
                                from openai.types.chat.chat_completion import Choice
                                from openai.types.chat.chat_completion_message import ChatCompletionMessage
                                
                                fallback_choice = Choice(
                                    index=index,
                                    message=ChatCompletionMessage(
                                        role="assistant",
                                        content=None,
                                        tool_calls=[mock_tool_call]
                                    ),
                                    finish_reason="tool_calls"
                                )
                                
                                choices.append(fallback_choice)
                                index += 1
                                continue  # Skip the normal processing for this candidate
                        
                        # For non-image requests, still raise the error to trigger retry
                        raise ValueError(f"Error in response data from LLM: {candidate.finish_message[:350]}...")
                    else:
                        raise ValueError(f"Error in response data from LLM: {response_data}")

                role = content.role
                assert role == "model", f"Unknown role in response: {role}"

                parts = content.parts

                # NOTE: we aren't properly supported multi-parts here anyways (we're just appending choices),
                #       so let's disable it for now

                # NOTE(Apr 9, 2025): there's a very strange bug on 2.5 where the response has a part with broken text
                # {'candidates': [{'content': {'parts': [{'functionCall': {'name': 'send_message', 'args': {'request_heartbeat': False, 'message': 'Hello! How can I make your day better?', 'inner_thoughts': 'User has initiated contact. Sending a greeting.'}}}], 'role': 'model'}, 'finishReason': 'STOP', 'avgLogprobs': -0.25891534213362066}], 'usageMetadata': {'promptTokenCount': 2493, 'candidatesTokenCount': 29, 'totalTokenCount': 2522, 'promptTokensDetails': [{'modality': 'TEXT', 'tokenCount': 2493}], 'candidatesTokensDetails': [{'modality': 'TEXT', 'tokenCount': 29}]}, 'modelVersion': 'gemini-1.5-pro-002'}
                # To patch this, if we have multiple parts we can take the last one
                if len(parts) > 1:
                    logger.warning(f"Unexpected multiple parts in response from Google AI: {parts}")
                    parts = [parts[-1]]

                # TODO support parts / multimodal
                # TODO support parallel tool calling natively
                # TODO Alternative here is to throw away everything else except for the first part
                for response_message in parts:
                    # Convert the actual message style to OpenAI style
                    if response_message.function_call:
                        function_call = response_message.function_call
                        function_name = function_call.name
                        function_args = function_call.args
                        assert isinstance(function_args, dict), function_args

                        # NOTE: this also involves stripping the inner monologue out of the function
                        if llm_config.put_inner_thoughts_in_kwargs:
                            from letta.local_llm.constants import INNER_THOUGHTS_KWARG_VERTEX

                            assert (
                                INNER_THOUGHTS_KWARG_VERTEX in function_args
                            ), f"Couldn't find inner thoughts in function args:\n{function_call}"
                            inner_thoughts = function_args.pop(INNER_THOUGHTS_KWARG_VERTEX)
                            assert inner_thoughts is not None, f"Expected non-null inner thoughts function arg:\n{function_call}"
                        else:
                            inner_thoughts = None

                        # Google AI API doesn't generate tool call IDs
                        openai_response_message = Message(
                            role="assistant",  # NOTE: "model" -> "assistant"
                            content=inner_thoughts,
                            tool_calls=[
                                ToolCall(
                                    id=get_tool_call_id(),
                                    type="function",
                                    function=FunctionCall(
                                        name=function_name,
                                        arguments=clean_json_string_extra_backslash(json_dumps(function_args)),
                                    ),
                                )
                            ],
                        )

                    else:
                        try:
                            # Structured output tool call
                            function_call = json_loads(response_message.text)
                            function_name = function_call["name"]
                            function_args = function_call["args"]
                            assert isinstance(function_args, dict), function_args

                            # NOTE: this also involves stripping the inner monologue out of the function
                            if llm_config.put_inner_thoughts_in_kwargs:
                                from letta.local_llm.constants import INNER_THOUGHTS_KWARG_VERTEX

                                assert (
                                    INNER_THOUGHTS_KWARG_VERTEX in function_args
                                ), f"Couldn't find inner thoughts in function args:\n{function_call}"
                                inner_thoughts = function_args.pop(INNER_THOUGHTS_KWARG_VERTEX)
                                assert inner_thoughts is not None, f"Expected non-null inner thoughts function arg:\n{function_call}"
                            else:
                                inner_thoughts = None

                            # Google AI API doesn't generate tool call IDs
                            openai_response_message = Message(
                                role="assistant",  # NOTE: "model" -> "assistant"
                                content=inner_thoughts,
                                tool_calls=[
                                    ToolCall(
                                        id=get_tool_call_id(),
                                        type="function",
                                        function=FunctionCall(
                                            name=function_name,
                                            arguments=clean_json_string_extra_backslash(json_dumps(function_args)),
                                        ),
                                    )
                                ],
                            )

                        except json.decoder.JSONDecodeError:
                            if candidate.finish_reason == "MAX_TOKENS":
                                raise ValueError(f"Could not parse response data from LLM: exceeded max token limit")
                            # Inner thoughts are the content by default
                            inner_thoughts = response_message.text

                            # Google AI API doesn't generate tool call IDs
                            openai_response_message = Message(
                                role="assistant",  # NOTE: "model" -> "assistant"
                                content=inner_thoughts,
                            )

                    # Google AI API uses different finish reason strings than OpenAI
                    # OpenAI: 'stop', 'length', 'function_call', 'content_filter', null
                    #   see: https://platform.openai.com/docs/guides/text-generation/chat-completions-api
                    # Google AI API: FINISH_REASON_UNSPECIFIED, STOP, MAX_TOKENS, SAFETY, RECITATION, OTHER
                    #   see: https://ai.google.dev/api/python/google/ai/generativelanguage/Candidate/FinishReason
                    finish_reason = candidate.finish_reason.value
                    if finish_reason == "STOP":
                        openai_finish_reason = (
                            "function_call"
                            if openai_response_message.tool_calls is not None and len(openai_response_message.tool_calls) > 0
                            else "stop"
                        )
                    elif finish_reason == "MAX_TOKENS":
                        openai_finish_reason = "length"
                    elif finish_reason == "SAFETY":
                        openai_finish_reason = "content_filter"
                    elif finish_reason == "RECITATION":
                        openai_finish_reason = "content_filter"
                    else:
                        raise ValueError(f"Unrecognized finish reason in Google AI response: {finish_reason}")

                    choices.append(
                        Choice(
                            finish_reason=openai_finish_reason,
                            index=index,
                            message=openai_response_message,
                        )
                    )
                    index += 1

            # if len(choices) > 1:
            #     raise UserWarning(f"Unexpected number of candidates in response (expected 1, got {len(choices)})")

            # NOTE: some of the Google AI APIs show UsageMetadata in the response, but it seems to not exist?
            #  "usageMetadata": {
            #     "promptTokenCount": 9,
            #     "candidatesTokenCount": 27,
            #     "totalTokenCount": 36
            #   }
            if response.usage_metadata:
                usage = UsageStatistics(
                    prompt_tokens=response.usage_metadata.prompt_token_count,
                    completion_tokens=response.usage_metadata.candidates_token_count,
                    total_tokens=response.usage_metadata.total_token_count,
                )
            else:
                # Count it ourselves
                assert input_messages is not None, f"Didn't get UsageMetadata from the API response, so input_messages is required"
                prompt_tokens = count_tokens(json_dumps(input_messages))  # NOTE: this is a very rough approximation
                completion_tokens = count_tokens(json_dumps(openai_response_message.model_dump()))  # NOTE: this is also approximate
                total_tokens = prompt_tokens + completion_tokens
                usage = UsageStatistics(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            response_id = str(uuid.uuid4())
            return ChatCompletionResponse(
                id=response_id,
                choices=choices,
                model=llm_config.model,  # NOTE: Google API doesn't pass back model in the response
                created=get_utc_time_int(),
                usage=usage,
            )
        except KeyError as e:
            raise e

    def get_function_call_response_schema(self, tool: dict) -> dict:
        return {
            "type": "OBJECT",
            "properties": {
                "name": {"type": "STRING", "enum": [tool["name"]]},
                "args": {
                    "type": "OBJECT",
                    "properties": tool["parameters"]["properties"],
                    "required": tool["parameters"]["required"],
                },
            },
            "propertyOrdering": ["name", "args"],
            "required": ["name", "args"],
        }

    @trace_method
    def handle_llm_error(self, e: Exception) -> Exception:
        """
        Maps Google/Gemini-specific errors to common LLMError types.
        Handles authentication, quota, context window, and API errors.
        """
        from letta.errors import (
            ContextWindowExceededError,
            ErrorCode,
            LLMAuthenticationError,
            LLMBadRequestError,
            LLMConnectionError,
            LLMError,
            LLMRateLimitError,
            LLMServerError,
            LLMTimeoutError,
        )

        error_str = str(e).lower()
        error_message = str(e)

        # Check for authentication/API key errors
        if any(keyword in error_str for keyword in ["api key", "authentication", "unauthorized", "invalid_api_key", "permission denied"]):
            logger.warning(f"[Google AI] Authentication error: {error_message}")
            return LLMAuthenticationError(
                message=f"Google AI authentication failed. Please check your GEMINI_API_KEY: {error_message}",
                code=ErrorCode.UNAUTHENTICATED,
                details={"original_error": error_message},
            )

        # Check for context window / request size errors
        if any(keyword in error_str for keyword in [
            "context length", "too large", "request too large", "maximum context",
            "exceeds", "token limit", "context window", "max_tokens"
        ]):
            logger.warning(f"[Google AI] Context window exceeded: {error_message}")
            return ContextWindowExceededError(
                message=f"Google AI request exceeded context window. Try reducing message history or tool count: {error_message}",
                details={"original_error": error_message},
            )

        # Check for rate limiting / quota errors
        if any(keyword in error_str for keyword in ["rate limit", "quota", "429", "resource exhausted"]):
            logger.warning(f"[Google AI] Rate limit or quota exceeded: {error_message}")
            return LLMRateLimitError(
                message=f"Google AI rate limit or quota exceeded: {error_message}",
                code=ErrorCode.RATE_LIMIT_EXCEEDED,
                details={"original_error": error_message},
            )

        # Check for invalid request / bad schema errors
        if any(keyword in error_str for keyword in [
            "invalid", "bad request", "400", "malformed", "unsupported",
            "schema", "parameter", "tool", "function"
        ]):
            logger.warning(f"[Google AI] Bad request (possibly invalid tool schema): {error_message}")
            return LLMBadRequestError(
                message=f"Google AI rejected request. This may be due to unsupported tool schema or invalid parameters: {error_message}",
                code=ErrorCode.INVALID_ARGUMENT,
                details={"original_error": error_message},
            )

        # Check for timeout errors
        if any(keyword in error_str for keyword in ["timeout", "timed out", "deadline"]):
            logger.warning(f"[Google AI] Request timeout: {error_message}")
            return LLMTimeoutError(
                message=f"Google AI request timed out: {error_message}",
                code=ErrorCode.TIMEOUT,
                details={"original_error": error_message},
            )

        # Check for server errors (5xx)
        if any(keyword in error_str for keyword in ["500", "502", "503", "504", "internal server error", "service unavailable"]):
            logger.warning(f"[Google AI] Server error: {error_message}")
            return LLMServerError(
                message=f"Google AI service error. This is a temporary issue, please retry: {error_message}",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                details={"original_error": error_message},
            )

        # Check for connection errors
        if any(keyword in error_str for keyword in ["connection", "network", "dns", "unreachable"]):
            logger.warning(f"[Google AI] Connection error: {error_message}")
            return LLMConnectionError(
                message=f"Failed to connect to Google AI: {error_message}",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                details={"original_error": error_message},
            )

        # Fallback to base implementation for unhandled errors
        logger.warning(f"[Google AI] Unhandled error type, using base handler: {error_message}")
        return super().handle_llm_error(e)
