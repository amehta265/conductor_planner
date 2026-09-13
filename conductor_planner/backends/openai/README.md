# OpenAI backend

Two entry points over one implementation.

## Hosted

```yaml
models:
  planner:
    kind: openai
    model: gpt-5
```

Key from `keys/openai.key`, `OPENAI_API_KEY`, or `api_key` / `api_key_file`.

## Any OpenAI-compatible endpoint

```yaml
models:
  grounder:
    kind: openai_compat
    model: BAAI/RoboBrain2.5-4B
    base_url: http://localhost:8001/v1
    require_key: false
```

This is the path for a local vLLM server, and the only reason the 16 GB card is
worth involving at all. See `docs/STRETCH_SETUP.md` for what actually fits.

`response_format: json_schema` and vLLM's `guided_json` are both sent when a
schema is passed; servers ignore the key they do not know.
