# CPU image for Phase 1. Cartpole / G1 stand / pick-and-place.
# GPU walk/reach still compile-and-block; do not put Isaac Sim in this file.
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml README.md docker-entrypoint.sh ./
COPY src ./src
COPY recipes ./recipes
COPY spec ./spec
COPY robots ./robots
COPY studio ./studio
RUN chmod +x docker-entrypoint.sh \
    && pip install --no-cache-dir -e ".[dev]"

ENV SDL_VIDEODRIVER=dummy
ENV SDL_AUDIODRIVER=dummy

ENTRYPOINT ["./docker-entrypoint.sh"]
