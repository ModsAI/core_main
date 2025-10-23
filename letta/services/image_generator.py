import asyncio
import base64
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from letta.log import get_logger
from letta.settings import model_settings

logger = get_logger(__name__)

class ImageGenerationService:
    """Service for generating images using Google's Gemini models"""
    
    def __init__(self):
        # Try different image generation models in order of preference
        # NOTE: Only certain models support image generation!
        self.models_to_try = [
            "gemini-2.0-flash-exp",  # Latest Gemini 2.0 with image generation
            "gemini-2.0-flash-exp-image-generation",  # Specific image generation model
            "imagen-3.0-generate-001",  # Google's Imagen model
        ]
        
        # Disable test mode to use real Gemini API (fallback to test images only if API fails)
        self.test_mode = False
        
        # Create storage directory for generated images (persistent location)
        self.storage_path = Path("/app/data/images")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Validate API key is available
        if not model_settings.gemini_api_key:
            logger.warning("Gemini API key not configured. Image generation will fail.")
    
    async def generate_image(self, prompt: str, agent_id: str, user_id: str) -> Dict:
        """Generate image using Gemini image generation model"""
        
        logger.info("=" * 80)
        logger.info("🎨 STARTING IMAGE GENERATION")
        logger.info("=" * 80)
        logger.info(f"   Agent: {agent_id}")
        logger.info(f"   User: {user_id}")
        logger.info(f"   Prompt: '{prompt}'")
        logger.info(f"   Models to try: {self.models_to_try}")
        logger.info(f"   Test mode: {self.test_mode}")
        logger.info(f"   Storage path: {self.storage_path}")
        logger.info(f"   API key configured: {bool(model_settings.gemini_api_key)}")
        if model_settings.gemini_api_key:
            logger.info(f"   API key length: {len(model_settings.gemini_api_key)}")
        logger.info("=" * 80)
        
        try:
            # Import Gemini client
            from google import genai
            from google.genai.types import HttpOptions
            
            # Validate API key
            if not model_settings.gemini_api_key:
                logger.error("❌ Gemini API key not configured!")
                raise ValueError("Gemini API key not configured. Please set GEMINI_API_KEY environment variable.")
            
            logger.info(f"✅ API key configured (length: {len(model_settings.gemini_api_key)})")
            
            # Create Gemini client with timeout
            client = genai.Client(
                api_key=model_settings.gemini_api_key,
                http_options=HttpOptions(timeout=30000)  # 30 second timeout
            )
            logger.info("✅ Gemini client configured")
            
            # Generate unique image ID
            image_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Try different models until one works
            response = None
            successful_model = None
            
            for i, model in enumerate(self.models_to_try, 1):
                try:
                    logger.info(f"🎯 Attempt {i}/{len(self.models_to_try)}: Trying model '{model}'")
                    
                    # Call Gemini image generation API with proper parameters
                    # Note: Using asyncio.to_thread to make the sync call async
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model=model,
                        contents=[{"parts": [{"text": prompt}]}],
                        config={
                            "responseModalities": ["Text", "Image"]
                        }
                    )
                    
                    logger.info(f"📡 Response received from '{model}':")
                    logger.info(f"   Response type: {type(response)}")
                    logger.info(f"   Has candidates: {hasattr(response, 'candidates')}")
                    
                    if response and hasattr(response, 'candidates') and response.candidates:
                        successful_model = model
                        logger.info(f"✅ Successfully got response from model: {model}")
                        logger.info(f"   Candidates count: {len(response.candidates)}")
                        break
                    else:
                        logger.warning(f"❌ Model {model} returned empty response")
                        
                except Exception as model_error:
                    logger.error(f"❌ Model '{model}' failed: {str(model_error)}")
                    logger.error(f"   Error type: {type(model_error)}")
                    continue
            
            if not response or not successful_model:
                raise ValueError("All image generation models failed")
            
            # Process the response
            if not response or not hasattr(response, 'candidates') or not response.candidates:
                raise ValueError("No image generated by Gemini API")
            
            # Extract image data from the first candidate
            candidate = response.candidates[0]
            logger.info(f"🔍 Processing candidate:")
            logger.info(f"   Candidate type: {type(candidate)}")
            logger.info(f"   Has content: {hasattr(candidate, 'content')}")
            
            # Check if the response contains image data
            if not hasattr(candidate, 'content') or not candidate.content:
                logger.error("❌ No content in Gemini response")
                raise ValueError("No content in Gemini response")
            
            # For image generation, the response should contain image data
            # The exact format may vary, so we'll handle multiple possibilities
            image_data = None
            image_url = None
            
            # Try to extract image data from response
            content = candidate.content
            logger.info(f"   Content type: {type(content)}")
            logger.info(f"   Has parts: {hasattr(content, 'parts')}")
            
            if hasattr(content, 'parts') and content.parts:
                logger.info(f"   Parts count: {len(content.parts)}")
                for i, part in enumerate(content.parts):
                    logger.info(f"   Part {i}: type={type(part)}")
                    logger.info(f"     Has inline_data: {hasattr(part, 'inline_data')}")
                    logger.info(f"     Has file_data: {hasattr(part, 'file_data')}")
                    logger.info(f"     Has text: {hasattr(part, 'text')}")
                    
                    # Check for inline image data
                    if hasattr(part, 'inline_data') and part.inline_data:
                        image_data = part.inline_data.data
                        logger.info(f"📷 Found inline image data: {len(image_data)} bytes")
                        break
                    elif hasattr(part, 'text') and part.text:
                        logger.info(f"📝 Found text: {part.text[:100]}...")
                    # Check for file data
                    elif hasattr(part, 'file_data') and part.file_data:
                        # Handle file-based response
                        image_url = part.file_data.file_uri
                        break
            
            # Save image if we have data
            image_path = None
            if image_data:
                # Save base64 image data to file
                image_filename = f"{image_id}.png"
                image_path = self.storage_path / image_filename
                
                try:
                    # Decode and save image with validation
                    decoded_data = base64.b64decode(image_data)
                    
                    # Validate we have actual image data (should be > 1KB for real images)
                    if len(decoded_data) < 1024:
                        logger.error(f"Image data too small ({len(decoded_data)} bytes), likely corrupted")
                        # Trigger fallback by raising exception
                        raise ValueError(f"Generated image data is too small ({len(decoded_data)} bytes) - triggering fallback")
                    
                    with open(image_path, "wb") as f:
                        f.write(decoded_data)
                    
                    logger.info(f"Image saved successfully to: {image_path} ({len(decoded_data)} bytes)")
                except Exception as e:
                    logger.error(f"Failed to save image: {str(e)}")
                    image_path = None
                    # If we have corrupted data and test mode is enabled, trigger fallback
                    if self.test_mode and "too small" in str(e):
                        logger.info("Triggering fallback test image generation due to corrupted API response")
                        raise e  # This will be caught by the outer exception handler and trigger fallback
            
            # Generate image URL for serving via HTTP endpoint
            image_filename = f"{image_id}.png" if image_path else None
            http_image_url = f"/images/{image_filename}" if image_filename else image_url
            
            # Prepare result
            result = {
                "image_id": image_id,
                "prompt": prompt,
                "timestamp": timestamp,
                "model": successful_model,
                "status": "success",
                "image_path": str(image_path) if image_path else None,
                "image_url": http_image_url,
                "message": f"✅ Image '{prompt}' generated successfully using {successful_model}\n🖼️ View image: {http_image_url}" if http_image_url else f"✅ Image '{prompt}' generated successfully using {successful_model}"
            }
            
            logger.info("🎯 FINAL RESULT - Image generation completed successfully!")
            logger.info(f"   Image ID: {image_id}")
            logger.info(f"   Status: {result.get('status')}")
            logger.info(f"   Image URL: {result.get('image_url')}")
            logger.info(f"   Image Path: {result.get('image_path')}")
            logger.info(f"   Model: {result.get('model')}")
            logger.info(f"   Message: {result.get('message')}")
            logger.info("=" * 80)
            return result
            
        except Exception as e:
            logger.error(f"🚨 Image generation failed: {str(e)}", exc_info=True)
            
            # If test mode is enabled, create a placeholder image
            if self.test_mode:
                logger.info(f"🔄 Test mode enabled, creating fallback test image...")
                try:
                    result = await self._create_test_image(prompt, agent_id, user_id)
                    logger.info(f"✅ Fallback test image created successfully!")
                    return result
                except Exception as test_error:
                    logger.error(f"❌ Test image creation also failed: {str(test_error)}")
            else:
                logger.info(f"⚠️ Test mode disabled, no fallback available")
            
            # Return error result
            return {
                "image_id": str(uuid.uuid4()),
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "model": "unknown",
                "status": "error",
                "error": str(e),
                "message": f"❌ Image generation failed: {str(e)}"
            }
    
    async def _create_test_image(self, prompt: str, agent_id: str, user_id: str) -> Dict:
        """Create a test image to verify the pipeline works"""
        logger.info(f"🎨 Creating test image for prompt: '{prompt}'")
        try:
            from PIL import Image, ImageDraw, ImageFont
            import textwrap
            
            # Generate unique image ID
            image_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Create a simple test image
            width, height = 512, 512
            image = Image.new('RGB', (width, height), color='lightblue')
            draw = ImageDraw.Draw(image)
            
            # Add text to the image
            try:
                # Try to use a default font
                font = ImageFont.load_default()
            except:
                font = None
            
            # Wrap the prompt text
            wrapped_text = textwrap.fill(f"Generated Image:\n{prompt}", width=40)
            
            # Draw the text
            text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            draw.text((x, y), wrapped_text, fill='black', font=font)
            
            # Add a border
            draw.rectangle([10, 10, width-10, height-10], outline='darkblue', width=3)
            
            # Save the image
            image_filename = f"{image_id}.png"
            image_path = self.storage_path / image_filename
            image.save(image_path, 'PNG')
            
            logger.info(f"Test image created successfully: {image_path}")
            
            # Generate image URL
            http_image_url = f"/images/{image_filename}"
            
            return {
                "image_id": image_id,
                "prompt": prompt,
                "timestamp": timestamp,
                "model": "test-image-generator",
                "status": "success",
                "image_path": str(image_path),
                "image_url": http_image_url,
                "message": f"✅ Test image '{prompt}' generated successfully!\n🖼️ View image: {http_image_url}"
            }
            
        except Exception as e:
            logger.error(f"Test image creation failed: {str(e)}")
            raise
    
    def cleanup_old_images(self, max_age_hours: int = 24):
        """Clean up old generated images to save disk space"""
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for image_file in self.storage_path.glob("*.png"):
                if current_time - image_file.stat().st_mtime > max_age_seconds:
                    image_file.unlink()
                    logger.info(f"Cleaned up old image: {image_file}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old images: {e}")