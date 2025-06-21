# Multi-stage build for AWS Cost Optimizer

# Stage 1: Build stage
FROM python:3.9-slim-bullseye AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install --user --no-cache-dir -e .

# Stage 2: Runtime stage
FROM python:3.9-slim-bullseye

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash optimizer

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/optimizer/.local

# Copy application code
COPY --from=builder /build /app

# Set environment variables
ENV PATH=/home/optimizer/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AWS_DEFAULT_REGION=us-east-1

# Create necessary directories
RUN mkdir -p /app/logs /app/output /app/config && \
    chown -R optimizer:optimizer /app

# Switch to non-root user
USER optimizer

# Expose port for web dashboard (if implemented)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["aws-cost-optimizer", "--help"]

# Example commands:
# docker run --rm ghcr.io/your-org/aws-cost-optimizer discover
# docker run --rm -v $(pwd)/config:/app/config ghcr.io/your-org/aws-cost-optimizer analyze