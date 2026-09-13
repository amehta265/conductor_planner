# Anthropic backend

Default provider for all three model roles.

```yaml
models:
  planner:
    kind: anthropic
    model: claude-sonnet-4-5
```

The key comes from `keys/anthropic.key`, `ANTHROPIC_API_KEY`, or an explicit
`api_key` / `api_key_file` in the model block. See `keys/README.md`.

## What this backend does that a plain HTTP call would not

**Forced tool call for structured output.** When the planner passes a JSON
schema, the request declares an `emit` tool and forces it, so the response is
schema-valid by construction rather than by luck. The planner's repair loop
then almost never fires — and every repair is a paid round trip.

**Prompt caching.** The system prompt is marked `cache_control: ephemeral`. It
is byte-identical across every turn of an episode, so you pay for it once. On a
30-step episode that is most of the input bill. This is why volatile memory
lives in the per-turn message and not in the system prompt — putting it in the
system prompt would invalidate the cache on every turn.

**Retry with backoff** on 429 and 529. An episode holds a physical robot in a
half-finished state; failing one over a transient overload is a bad trade.

## Watching the bill

```python
backend.cost_report()
# anthropic:claude-sonnet-4-5: 412030 input tokens (94% served from cache), 8210 output
```

If the cached percentage is low, something is mutating the system prompt between
turns. That is a bug, not a tuning parameter.
