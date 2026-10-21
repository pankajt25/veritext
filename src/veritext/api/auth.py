"""API key authentication, rate limiting, and webhook signing for Veritext REST API."""

from collections import deque
from datetime import datetime, timezone
import hashlib
import hmac
import json
import logging
from pathlib import Path
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_KEYS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "config"
    / "api_keys.json"
)

# In-memory sliding window rate limiter: key_hash -> deque of timestamps
RATE_LIMIT_WINDOWS: Dict[str, deque] = {}
RATE_LIMIT_MAX_REQUESTS = 60  # max requests per minute
RATE_LIMIT_WINDOW_SECONDS = 60


def hash_key(key: str) -> str:
    """Returns SHA-256 hash of API key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def load_api_keys(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads hashed API keys from configuration file."""
    path = Path(config_path) if config_path else DEFAULT_KEYS_PATH
    if not path.is_file():
        return {"keys": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"keys": {}}
    except Exception as e:
        logger.error(f"Failed to read API keys from {path}: {e}")
        return {"keys": {}}


def save_api_keys(keys_data: Dict[str, Any], config_path: Optional[Union[str, Path]] = None):
    """Saves API keys dictionary to configuration file."""
    path = Path(config_path) if config_path else DEFAULT_KEYS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(keys_data, indent=2), encoding="utf-8")


def create_api_key(name: str, config_path: Optional[Union[str, Path]] = None) -> Tuple[str, str]:
    """Generates a new API key, stores its hash, and returns (plaintext_key, key_hash)."""
    raw_token = secrets.token_hex(24)
    api_key = f"vt_live_{raw_token}"
    k_hash = hash_key(api_key)

    data = load_api_keys(config_path)
    keys_map = data.setdefault("keys", {})
    keys_map[k_hash] = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active": True,
        "prefix": api_key[:12] + "...",
    }
    save_api_keys(data, config_path)
    logger.info(f"Generated API key '{name}' (prefix: {api_key[:12]}...)")
    return api_key, k_hash


def revoke_api_key(name_or_prefix: str, config_path: Optional[Union[str, Path]] = None) -> bool:
    """Revokes an API key matching the given name or prefix."""
    data = load_api_keys(config_path)
    keys_map = data.get("keys", {})
    found = False
    for k_hash, meta in list(keys_map.items()):
        if meta.get("name") == name_or_prefix or meta.get("prefix", "").startswith(name_or_prefix):
            meta["active"] = False
            found = True
            break
    if found:
        save_api_keys(data, config_path)
    return found


def verify_api_key(key: str, config_path: Optional[Union[str, Path]] = None) -> Optional[Dict[str, Any]]:
    """Validates an incoming plaintext API key against stored hashes."""
    if not key or not key.startswith("vt_live_"):
        return None
    k_hash = hash_key(key)
    data = load_api_keys(config_path)
    keys_map = data.get("keys", {})
    meta = keys_map.get(k_hash)
    if meta and meta.get("active", True):
        return meta
    return None


def check_rate_limit(
    key_hash: str,
    max_requests: int = RATE_LIMIT_MAX_REQUESTS,
    window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
) -> bool:
    """Checks whether the client has exceeded the sliding window rate limit.

    Returns:
        True if request is allowed, False if rate limited.
    """
    now = time.time()
    if key_hash not in RATE_LIMIT_WINDOWS:
        RATE_LIMIT_WINDOWS[key_hash] = deque()

    q = RATE_LIMIT_WINDOWS[key_hash]
    # Purge old timestamps
    cutoff = now - window_seconds
    while q and q[0] < cutoff:
        q.popleft()

    if len(q) >= max_requests:
        return False

    q.append(now)
    return True


def sign_webhook_payload(secret: str, payload_bytes: bytes) -> str:
    """Computes HMAC-SHA256 signature for webhook notifications."""
    sig = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    return f"sha256={sig}"
