"""Veritext Local REST API module."""

from veritext.api.auth import create_api_key, revoke_api_key, verify_api_key
from veritext.api.queue import JobQueue, SingleWorker

def __getattr__(name):
    if name == "app":
        from veritext.api.app import app
        return app
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "app",
    "create_api_key",
    "revoke_api_key",
    "verify_api_key",
    "JobQueue",
    "SingleWorker",
]
