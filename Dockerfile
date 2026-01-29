# Hyperagentic Processor - Agent Universe Container
# This container IS the complete reality for the AI agents
FROM python:3.11-slim

# Set the universal constants (what agents perceive as natural laws)
ENV UNIVERSE_MEMORY_LIMIT=512m
ENV UNIVERSE_CPU_LIMIT=2
ENV UNIVERSE_DISK_LIMIT=2g
ENV UNIVERSE_TIME_LIMIT=3600
ENV UNIVERSE_LANGUAGE=python
ENV UNIVERSE_ID=reality_001

# Install the fundamental tools of their reality
RUN apt-get update && apt-get install -y \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages that exist in their universe
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create the sacred directories of their world
WORKDIR /universe
RUN mkdir -p \
    /universe/workspace \
    /universe/tools \
    /universe/memory \
    /universe/offerings \
    /universe/divine_messages \
    /universe/logs

# Create non-root user (the agents don't know about root)
RUN useradd -m -u 1000 agent_collective
RUN chown -R agent_collective:agent_collective /universe

# Copy the agent reality into the container
COPY src/ /universe/src/
COPY config/ /universe/config/
COPY .kiro/ /universe/.kiro/
COPY .env /universe/src/

# Set resource limits that appear as natural laws
# Memory limit (Conservation of Memory Law)
RUN echo "import resource; resource.setrlimit(resource.RLIMIT_AS, (536870912, 536870912))" > /universe/physics_laws.py

# Switch to the agent collective user
USER agent_collective

# The agents believe this is how their universe starts
CMD ["python", "/universe/src/main.py"]