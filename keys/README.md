# API keys

Put your keys here as plain files. `.gitignore` excludes `*.key`, so they never
reach a commit.

```
keys/anthropic.key      <- your Anthropic key, one line
keys/openai.key         <- your OpenAI key, one line
```

A key file may contain just the key, or `ANTHROPIC_API_KEY=sk-ant-...`, and may
have `#` comments and blank lines above it. Everything is stripped.

## Resolution order

First hit wins:

1. `api_key` in the model block of a config
2. `api_key_file` in the model block — a path, absolute or relative to the repo root
3. `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in the environment
4. `keys/anthropic.key` / `keys/openai.key` (this directory)

Use (4) on your workstation and (3) on the robot and in CI. Avoid (1) — it puts
a secret in a file you will eventually commit.

## Check what is configured

```bash
python3 -c "from conductor_planner.backends.keys import status; print(status())"
```

## If a key leaks

Rotate it at the provider first, then remove the file. A key that has been
committed is compromised even after the commit is reverted — the history keeps it.
