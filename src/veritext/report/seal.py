"""Tamper-evident cryptographically sealed reports using canonical JSON and HMAC-SHA256.

Detects modifications by third parties without access to the local signing key.
Key is stored locally at data/config/report_key with strict 0600 file permissions.
Provides verification API and CLI: python -m veritext.report verify <report.json>.

Disclaimer: Cryptographic verification demonstrates file integrity and detects edits made
without the private signing key. It does not constitute legal proof in a court of law.
"""

from datetime import datetime, timezone
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import stat
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_KEY_PATH = Path("data/config/report_key")
SEAL_NOTICE = (
    "Tamper-evident seal detects unauthorized edits by parties without the local signing key. "
    "Demonstrates file integrity; does not constitute formal legal proof."
)


def get_or_create_signing_key(key_path: Optional[Union[str, Path]] = None) -> bytes:
    """Retrieves or initializes the 256-bit local HMAC signing key.

    Enforces 0600 (owner read/write only) file permissions on Unix-like filesystems.
    """
    path = Path(key_path) if key_path else DEFAULT_KEY_PATH

    if path.is_file():
        # Enforce 0600 permissions
        try:
            mode = os.stat(path).st_mode
            if mode & 0o077:  # group or other permissions present
                os.chmod(path, 0o600)
        except Exception as e:
            logger.warning(f"Could not check/set permissions on {path}: {e}")

        key_data = path.read_bytes()
        if len(key_data) == 32:
            return key_data
        elif len(key_data) == 64:  # Hex-encoded string
            try:
                return bytes.fromhex(key_data.decode("ascii").strip())
            except ValueError:
                pass
        # If key length is unexpected, re-pad or derive key
        return hashlib.sha256(key_data).digest()

    # Generate fresh 32-byte (256-bit) cryptographically strong random key
    path.parent.mkdir(parents=True, exist_ok=True)
    new_key = os.urandom(32)

    # Write securely with 0600 permissions
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    mode = stat.S_IRUSR | stat.S_IWUSR  # 0600
    try:
        fd = os.open(str(path), flags, mode)
        with open(fd, "wb") as f:
            f.write(new_key)
        os.chmod(path, 0o600)
    except Exception:
        # Fallback normal write and chmod
        path.write_bytes(new_key)
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass

    return new_key


def compute_canonical_json(data: Dict[str, Any]) -> bytes:
    """Produces deterministic canonical UTF-8 JSON bytes with sorted keys and no whitespace.

    Excludes any existing 'seal' dictionary to ensure consistent hash derivation.
    """
    # Create shallow copy omitting 'seal'
    cleaned = {k: v for k, v in data.items() if k != "seal"}
    return json.dumps(
        cleaned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def compute_seal(
    data: Dict[str, Any],
    key: Optional[bytes] = None,
    key_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Generates the cryptographic seal block for a report dictionary."""
    canonical_bytes = compute_canonical_json(data)
    canonical_sha256 = hashlib.sha256(canonical_bytes).hexdigest()

    signing_key = key if key is not None else get_or_create_signing_key(key_path)
    sig = hmac.new(signing_key, canonical_bytes, hashlib.sha256).hexdigest()

    return {
        "version": "1.0",
        "algorithm": "HMAC-SHA256",
        "canonical_sha256": canonical_sha256,
        "signature": sig,
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "notice": SEAL_NOTICE,
    }


def seal_report_dict(
    data: Dict[str, Any],
    key: Optional[bytes] = None,
    key_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Applies a tamper-evident seal to a report dictionary in-place and returns it."""
    seal = compute_seal(data, key=key, key_path=key_path)
    data["seal"] = seal
    return data


def verify_report_seal(
    data: Dict[str, Any],
    key: Optional[bytes] = None,
    key_path: Optional[Union[str, Path]] = None,
) -> Tuple[bool, str]:
    """Verifies the integrity and authenticity of a sealed report dictionary.

    Returns:
        (is_valid, reason_message)
    """
    seal = data.get("seal")
    if not isinstance(seal, dict):
        return False, "Report contains no tamper-evident cryptographic seal block."

    algo = seal.get("algorithm")
    if algo != "HMAC-SHA256":
        return False, f"Unsupported seal algorithm '{algo}' (expected 'HMAC-SHA256')."

    stored_sha = seal.get("canonical_sha256")
    stored_sig = seal.get("signature")

    if not stored_sha or not stored_sig:
        return False, "Seal block is missing canonical_sha256 or signature fields."

    canonical_bytes = compute_canonical_json(data)
    computed_sha = hashlib.sha256(canonical_bytes).hexdigest()

    if not hmac.compare_digest(computed_sha, stored_sha):
        return (
            False,
            f"Content checksum mismatch (computed: {computed_sha[:16]}..., expected: {stored_sha[:16]}...). "
            f"Document payload has been modified since it was sealed.",
        )

    signing_key = key if key is not None else get_or_create_signing_key(key_path)
    expected_sig = hmac.new(signing_key, canonical_bytes, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(expected_sig, stored_sig):
        return (
            False,
            "HMAC signature mismatch. The report was modified or sealed with a different signing key.",
        )

    return True, "Report cryptographic seal is VALID. Signature matches local key and content is intact."


def verify_file(
    file_path: Union[str, Path],
    key: Optional[bytes] = None,
    key_path: Optional[Union[str, Path]] = None,
) -> Tuple[bool, str]:
    """Verifies a report JSON file from disk."""
    path = Path(file_path)
    if not path.is_file():
        return False, f"File not found: '{file_path}'"

    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Invalid JSON format in '{file_path}': {e}"

    if not isinstance(content, dict):
        return False, f"Root element of '{file_path}' must be a JSON object."

    return verify_report_seal(content, key=key, key_path=key_path)
