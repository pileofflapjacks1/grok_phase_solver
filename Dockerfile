# grok-phase-solver — lightweight scientific Python image
# Build:  docker build -t grok-phase-solver:0.6.0 .
# Run:    docker run --rm -v "$PWD:/data" grok-phase-solver:0.6.0 \
#           gps-solve --hkl /data/demo.hkl --ins /data/demo.ins --out /data/out

FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="grok-phase-solver" \
      org.opencontainers.image.description="Open physics/AI X-ray phasing assistant" \
      org.opencontainers.image.source="https://github.com/pileofflapjacks1/grok_phase_solver" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY docs ./docs
COPY data/processed ./data/processed

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -e ".[gui]" \
    && pip install --no-cache-dir plotly

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["gps-solve"]
CMD ["--help"]
