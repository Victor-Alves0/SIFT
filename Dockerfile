# SIFT as an OpenAPI tool server (for OpenWebUI / REST clients).
# Customize examples/serve_http.py with your own tools, then build & run:
#   docker build -t sift-server .
#   docker run -p 8000:8000 -e SIFT_API_KEY=secret sift-server
FROM python:3.12-slim

WORKDIR /app

# install deps first for better layer caching
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[server]"

# your tool definitions / importers live here
COPY examples ./examples

EXPOSE 8000
ENV SIFT_HOST=0.0.0.0 SIFT_PORT=8000
CMD ["python", "examples/serve_http.py"]
