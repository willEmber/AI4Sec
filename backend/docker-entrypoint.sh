#!/bin/sh
set -e

# dockerd creates a missing bind-mount source as root:root, so the unprivileged
# runtime user cannot write the data dir without this fix-up. Run as root only
# long enough to prepare/chown the data dir, then drop to the `scholar` user.
DATA_DIR="${DATA_DIR:-/scholar/backend/data}"
mkdir -p "$DATA_DIR"

if [ "$(id -u)" = "0" ]; then
    chown -R scholar:scholar "$DATA_DIR" 2>/dev/null || true
    exec gosu scholar "$@"
fi

exec "$@"
