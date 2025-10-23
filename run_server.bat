@echo off
docker run -it --rm -p 8283:8283 --env-file .env -v "%cd%":/app -w /app letta_server:latest poetry run letta server --port=8283 