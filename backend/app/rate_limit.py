from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

# Shared rate limiter instance — imported by main.py and all routers.
# Kept in a separate module to avoid circular imports between main ↔ routers.
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
