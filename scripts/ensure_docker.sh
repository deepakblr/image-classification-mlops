#!/usr/bin/env bash
# Ensure a Docker daemon is reachable (Colima on macOS, or any existing daemon).

ensure_docker() {
  export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"

  if docker info >/dev/null 2>&1; then
    echo "Docker already available"
  else
    if command -v colima >/dev/null 2>&1; then
      export DOCKER_HOST="${DOCKER_HOST:-unix://${HOME}/.colima/default/docker.sock}"
      if ! colima status >/dev/null 2>&1; then
        echo "Starting Colima (Docker runtime)..."
        colima start --cpu "${COLIMA_CPU:-4}" --memory "${COLIMA_MEMORY:-6}"
      fi
      if docker context ls --format '{{.Name}}' 2>/dev/null | grep -qx colima; then
        docker context use colima >/dev/null
      fi
    fi

    if ! docker info >/dev/null 2>&1; then
      echo "ERROR: Docker daemon is not running." >&2
      echo "Start Colima (colima start) or Docker Desktop, then retry." >&2
      return 1
    fi
    echo "Docker is ready"
  fi
}

# Prefer Compose V2 plugin; fall back to standalone docker-compose (Homebrew).
docker_compose() {
  export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
  else
    echo "ERROR: neither 'docker compose' nor 'docker-compose' is installed." >&2
    return 1
  fi
}
