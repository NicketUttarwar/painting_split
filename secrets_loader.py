"""
Load secrets from secrets.yaml (gitignored). Never commit secrets.yaml.
Use secrets.example.yaml as a template, then copy to secrets.yaml.
"""
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
SECRETS_FILE = PROJECT_ROOT / "secrets.yaml"
_cached: Optional[dict[str, Any]] = None


def load_secrets() -> dict[str, Any]:
    """Load secrets from secrets.yaml. Returns {} if file missing or invalid."""
    global _cached
    if _cached is not None:
        return _cached
    if not SECRETS_FILE.is_file():
        _cached = {}
        return _cached
    try:
        import yaml
        with open(SECRETS_FILE, encoding="utf-8") as f:
            _cached = yaml.safe_load(f) or {}
        return _cached
    except Exception:
        _cached = {}
        return _cached


def get_openai_api_key() -> Optional[str]:
    """Return OpenAI API key from secrets, or None if not set."""
    secrets = load_secrets()
    openai_cfg = secrets.get("openai") or {}
    key = openai_cfg.get("api_key")
    if isinstance(key, str) and key.strip() and "your-" not in key.lower():
        return key.strip()
    return None
