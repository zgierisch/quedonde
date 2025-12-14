# Session Checkpoint LLM Playbook

This guide explains how humans and LLM copilots coordinate when capturing or replaying quedonde session checkpoints. Pair it with the canonical schema in [../specs/session_state_checkpointing.md](../specs/session_state_checkpointing.md) whenever you need authoritative field definitions.

## Prerequisites
- Local checkout already indexed via `python quedonde.py migrate` and `python quedonde.py index`.
- `quedonde.py session dump/resume` available in the current workspace (direct script call or editable install).
- Agreement on which subsystems, files, and symbols matter for the investigation. Treat these lists as the contract between operators and the LLM.

## Manual Workflow
1. **Frame the scope** – enumerate subsystems, file paths, and symbols that capture the investigation. The scope lists become part of the checkpoint and must stay concise (≤64 character subsystem names, relative file paths).
2. **Emit a checkpoint** – run `python quedonde.py session dump ...` with the scoped arguments.
3. **Review drift warnings** – when resuming, the CLI automatically revalidates structural versions, source paths, and fan-in/out counts. Investigate any warning before trusting the data.
4. **Share sanitized JSON** – the schema forbids raw code blocks >120 characters; nevertheless, skim the JSON before passing it to the LLM to ensure no sensitive payload slipped in.
5. **Update decisions/questions** – use repeated `--decision` and `--question` flags to capture conclusions and open items right away. This avoids stale notes getting lost between runs.

## Batch Automation
For multi-subsystem captures, feed a JSON config into `scripts/session_checkpoint.py` (added in Phase 3) to avoid typing dozens of flags.

```json
{
  "output_dir": "checkpoints",
  "default_confidence": "medium",
  "sessions": [
    {
      "session_id": "climate_stage2",
      "subsystems": ["Climate"],
      "files": ["src/climate/controller.py"],
      "symbols": ["ClimateController", "MoistureBridge"],
      "decisions": ["Delay refactor::still high fan-in"],
      "questions": ["Is MoistureBridge still needed post-uplift?"],
      "confidence": "high",
      "append": true
    }
  ]
}
```

Run it with:

```powershell
python scripts/session_checkpoint.py --config checkpoints/climate_plan.json
```

The helper expands each entry into a `session dump` invocation, applies consistent dependency limits, and mirrors the new scope into `checkpoints/<session_id>.json` unless you override `output` per session.

## LLM Collaboration Prompts
Use structured prompts so the LLM can acknowledge when a checkpoint exists versus when it needs fresh context.

**Requesting a checkpoint**
```
You are my structural assistant. Please run `python quedonde.py session dump \
  --session-id climate_stage2 \
  --subsystem Climate \
  --file src/climate/controller.py \
  --symbol ClimateController \
  --symbol MoistureBridge \
  --decision "Delay refactor::still high fan-in" \
  --question "Is MoistureBridge still needed post-uplift?" \
  --output checkpoints/climate.json`.
Confirm once the JSON file is written.
```

**Resuming from a checkpoint**
```
Load `checkpoints/climate.json`, session `climate_stage2`, run `python quedonde.py session resume --input checkpoints/climate.json --session-id climate_stage2 --json`, and summarize:
1. Structural version
2. Symbols plus fan-in/out counts
3. Outstanding questions
Call out any warnings printed to stderr.
```

## Guardrails & Reminders
- Never skip verification unless the workflow explicitly allows drift (`--skip-verify` should remain rare and well-documented).
- Keep `source_paths` lists short; if a symbol appears in more than five files, reduce the scope until the investigation stabilizes.
- Store checkpoints under an ignored directory such as `checkpoints/`; do not commit them.
- When the schema version changes, regenerate checkpoints immediately—old files will fail validation until migrated.
