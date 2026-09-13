"""API key resolution.

One rule, in one place, so a missing key produces a sentence that tells you
what to do rather than a 401 three layers down.

Resolution order, first hit wins:

  1. `api_key` passed explicitly (config or constructor)
  2. `api_key_file` passed explicitly -- a path to a file containing the key
  3. the provider's environment variable (ANTHROPIC_API_KEY / OPENAI_API_KEY)
  4. `keys/<provider>.key` next to the repo root

Option 4 is the one to use day to day: drop the key in a file, never paste it
into a config, and `.gitignore` already excludes `keys/*.key`. Option 3 is what
you want on the robot and in CI.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "sk-proj-wIHKsIvT8anUJztMb4QAT3BlbkFJvgtmQ8WYi5Nn3ja6cPAY",
}


class MissingKeyError(RuntimeError):
    pass


def repo_root() -> Path:
    """The directory holding `keys/`, found by walking up from this file."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "keys").is_dir() or (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[2]


def key_dir() -> Path:
    return repo_root() / "keys"


def resolve(
    provider: str,
    api_key: str | None = None,
    api_key_file: str | Path | None = None,
    required: bool = True,
) -> str | None:
    """Return the key for `provider`, or raise with instructions."""
    provider = provider.lower()
    if api_key:
        return api_key.strip()

    if api_key_file:
        p = Path(api_key_file).expanduser()
        if not p.is_absolute():
            p = repo_root() / p
        if not p.is_file():
            raise MissingKeyError(f"api_key_file points at {p}, which does not exist")
        return _read(p)

    env_name = ENV_VARS.get(provider, f"{provider.upper()}_API_KEY")
    if os.environ.get(env_name):
        return os.environ[env_name].strip()

    candidate = key_dir() / f"{provider}.key"
    if candidate.is_file():
        return _read(candidate)

    if not required:
        return None
    raise MissingKeyError(
        f"no API key found for {provider}. Do one of:\n"
        f"  - write the key into {candidate}\n"
        f"  - export {env_name}=...\n"
        f"  - set api_key or api_key_file in the model block of your config\n"
        f"See keys/README.md."
    )


def _read(p: Path) -> str:
    """Read a key file, tolerating comments, blank lines and KEY=value form."""
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line and not line.startswith("sk-"):
            line = line.split("=", 1)[1].strip()
        return line.strip().strip('"').strip("'")
    raise MissingKeyError(f"{p} exists but contains no key")


def status() -> dict[str, str]:
    """Where each provider's key would come from right now. For diagnostics."""
    out: dict[str, str] = {}
    for provider, env_name in ENV_VARS.items():
        if os.environ.get(env_name):
            out[provider] = f"environment variable {env_name}"
        elif (key_dir() / f"{provider}.key").is_file():
            out[provider] = str(key_dir() / f"{provider}.key")
        else:
            out[provider] = "NOT CONFIGURED"
    return out
