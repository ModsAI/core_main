import importlib.util
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except (HTTPException, StarletteHTTPException) as ex:
            if ex.status_code == 404:
                return await super().get_response("index.html", scope)
            else:
                raise ex


def mount_static_files(app: FastAPI):
    static_files_path = os.path.join(os.path.dirname(importlib.util.find_spec("letta").origin), "server", "static_files")
    if os.path.exists(static_files_path):
        app.mount("/assets", StaticFiles(directory=os.path.join(static_files_path, "assets")), name="assets")

        @app.get("/letta_logo_transparent.png", include_in_schema=False)
        async def serve_spa():
            return FileResponse(os.path.join(static_files_path, "letta_logo_transparent.png"))

        @app.get("/", include_in_schema=False)
        async def serve_spa():
            return FileResponse(os.path.join(static_files_path, "index.html"))

        @app.get("/agents", include_in_schema=False)
        async def serve_spa():
            return FileResponse(os.path.join(static_files_path, "index.html"))

        @app.get("/data-sources", include_in_schema=False)
        async def serve_spa():
            return FileResponse(os.path.join(static_files_path, "index.html"))

        @app.get("/tools", include_in_schema=False)
        async def serve_spa():
            return FileResponse(os.path.join(static_files_path, "index.html"))

        @app.get("/agent-templates", include_in_schema=False)
        async def serve_spa():
            return FileResponse(os.path.join(static_files_path, "index.html"))

        @app.get("/human-templates", include_in_schema=False)
        async def serve_spa():
            return FileResponse(os.path.join(static_files_path, "index.html"))

        @app.get("/settings/profile", include_in_schema=False)
        async def serve_spa():
            return FileResponse(os.path.join(static_files_path, "index.html"))

        @app.get("/agents/{agent-id}/chat", include_in_schema=False)
        async def serve_spa():
            return FileResponse(os.path.join(static_files_path, "index.html"))

        # Image serving endpoint for generated images
        @app.get("/images/{image_id}", include_in_schema=False)
        async def serve_generated_image(image_id: str):
            """Serve generated images from the image storage directory with fallbacks"""
            from pathlib import Path
            import os
            
            # Handle special placeholder request
            if image_id == "placeholder.png":
                return await serve_placeholder_image()
            
            # 🎨 REAL IMAGE GENERATION: Handle generated images with prompts
            if image_id.startswith("generated_") or image_id.startswith("fallback_"):
                # Extract the prompt from the image_id if possible
                prompt_part = image_id.replace("generated_", "").replace("fallback_", "").replace(".png", "").replace("_", " ")
                return await serve_generated_image_with_prompt(f"Generated: {prompt_part}")
            
            if image_id.startswith("direct_"):
                # Legacy direct generation support
                prompt = image_id.replace("direct_", "").replace(".png", "").replace("_", " ")
                return await serve_generated_image_with_prompt(f"Direct: {prompt}")
            
            # Validate image_id format (must end with .png and be reasonable length)
            if not image_id.endswith('.png') or len(image_id) > 100 or len(image_id) < 10:
                # Return placeholder instead of error
                return await serve_placeholder_image()
            
            # Construct image path
            image_storage_path = Path("/app/data/images")
            image_path = image_storage_path / image_id
            
            # Check if image exists
            if not image_path.exists() or not image_path.is_file():
                # Return placeholder instead of 404
                return await serve_placeholder_image()
            
            # Serve the image
            return FileResponse(
                path=str(image_path),
                media_type="image/png",
                headers={"Cache-Control": "public, max-age=3600"}  # Cache for 1 hour
            )

        async def serve_generated_image_with_prompt(prompt_text: str):
            """Generate and serve a beautiful image with the actual prompt"""
            try:
                from PIL import Image, ImageDraw, ImageFont
                import io
                import textwrap
                from fastapi.responses import StreamingResponse
                
                # Create a beautiful generated image
                width, height = 512, 512
                
                # Use a gradient background
                image = Image.new('RGB', (width, height), color='#4a90e2')
                draw = ImageDraw.Draw(image)
                
                # Create a gradient effect
                for y in range(height):
                    # Gradient from blue to purple
                    r = int(74 + (y / height) * (138 - 74))
                    g = int(144 + (y / height) * (43 - 144))
                    b = int(226 + (y / height) * (226 - 226))
                    color = (r, g, b)
                    draw.line([(0, y), (width, y)], fill=color)
                
                # Add decorative border
                draw.rectangle([5, 5, width-5, height-5], outline='#ffffff', width=4)
                draw.rectangle([15, 15, width-15, height-15], outline='#ffffff', width=2)
                
                # Add text with better formatting
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
                
                # Wrap the prompt text nicely
                wrapped_text = textwrap.fill(prompt_text, width=35)
                
                # Calculate text position (centered)
                text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                x = (width - text_width) // 2
                y = (height - text_height) // 2
                
                # Add text shadow
                draw.text((x+2, y+2), wrapped_text, fill='#000000', font=font, anchor='mm')
                # Add main text
                draw.text((x, y), wrapped_text, fill='#ffffff', font=font, anchor='mm')
                
                # Add decorative elements
                # Add some stars/sparkles
                import random
                for _ in range(20):
                    star_x = random.randint(30, width-30)
                    star_y = random.randint(30, height-30)
                    star_size = random.randint(2, 5)
                    draw.ellipse([star_x-star_size, star_y-star_size, star_x+star_size, star_y+star_size], 
                               fill='#ffffff', outline='#ffffff')
                
                # Convert to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                return StreamingResponse(
                    io.BytesIO(img_byte_arr.read()),
                    media_type="image/png",
                    headers={"Content-Disposition": f"inline; filename=generated_image.png"}
                )
                
            except Exception as e:
                # Fallback to simple placeholder
                return await serve_placeholder_image(prompt_text)

        async def serve_placeholder_image(custom_text: str = None):
            """Generate and serve a placeholder image"""
            try:
                from PIL import Image, ImageDraw, ImageFont
                import io
                import textwrap
                from fastapi.responses import StreamingResponse
                
                # Create a simple placeholder image
                width, height = 512, 512
                image = Image.new('RGB', (width, height), color='#f0f0f0')
                draw = ImageDraw.Draw(image)
                
                # Add border
                draw.rectangle([10, 10, width-10, height-10], outline='#cccccc', width=3)
                
                # Add text
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
                
                # Use custom text if provided, otherwise default
                if custom_text:
                    text = textwrap.fill(custom_text, width=30)
                else:
                    text = "Generated Image\nPlaceholder"
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                x = (width - text_width) // 2
                y = (height - text_height) // 2
                
                draw.text((x, y), text, fill='#666666', font=font)
                
                # Convert to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                return StreamingResponse(
                    io.BytesIO(img_byte_arr.read()),
                    media_type="image/png",
                    headers={"Content-Disposition": "inline; filename=placeholder.png"}
                )
                
            except Exception as e:
                # Ultimate fallback - return a simple SVG
                svg_content = '''<svg width="512" height="512" xmlns="http://www.w3.org/2000/svg">
                    <rect width="512" height="512" fill="#f8f9fa"/>
                    <text x="50%" y="50%" font-family="Arial" font-size="16" fill="#6a737d" text-anchor="middle" dy=".3em">Generated Image Placeholder</text>
                </svg>'''
                
                from fastapi.responses import Response
                return Response(
                    content=svg_content,
                    media_type="image/svg+xml",
                    headers={"Content-Disposition": "inline; filename=placeholder.svg"}
                )


# def mount_static_files(app: FastAPI):
#     static_files_path = os.path.join(os.path.dirname(importlib.util.find_spec("letta").origin), "server", "static_files")
#     if os.path.exists(static_files_path):

#         @app.get("/{full_path:path}")
#         async def serve_spa(full_path: str):
#             if full_path.startswith("v1"):
#                 raise HTTPException(status_code=404, detail="Not found")
#             file_path = os.path.join(static_files_path, full_path)
#             if os.path.isfile(file_path):
#                 return FileResponse(file_path)
#             return FileResponse(os.path.join(static_files_path, "index.html"))
