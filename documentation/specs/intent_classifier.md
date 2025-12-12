# Intent Classifier Rules

| Intent | Query patterns | Handler |
| --- | --- | --- |
| `find_symbol` | contains `where is`, `find symbol`, `definition of`, or explicit `symbol:` prefix | Looks up symbol rows in the `symbols` table and returns path/kind/line spans. |
| `callers` | contains `who calls`, `callers of`, `referenced by`, or `incoming` | Reads `edges` with `relation='calls'` where `dst_symbol` matches the requested name. |
| `deps` | contains `depends on`, `dependencies`, `what uses`, or `outgoing` | Reads `edges` for the requested symbol as `src_symbol` across `calls`, `imports`, `includes`, and `references`. |
| `explain` | contains `explain`, `why does`, `annotations for`, `describe` | Combines symbol spans, annotations, incoming callers, and outgoing dependencies into a single report. |

All heuristics run in priority order: `find_symbol` keywords win over others; otherwise the first matching rule is chosen. Queries without any matches fall back to lexical search.

## Expanded heuristics

- Additional callers cues: `call tree`, `incoming references`, `who invokes`, and `trace to`.
- Additional dependency cues: `what does <symbol> use`, `who depends on`, `downstream`, `imports for`, `fan-out`.
- Additional explain cues: `tell me about`, `give context for`, `annotate`. These require only a descriptive verb plus a detected symbol.
- Stop-word filter ignores helper verbs/pronouns (`what`, `is`, `does`, etc.) so the last meaningful token becomes the candidate symbol (see `documentation/fixtures/intent/queries.md`).

## Ask workflow

1. Run migrations (`python quedonde.py migrate`) and re-index (`python quedonde.py index`) to ensure structural tables are populated.
2. Invoke `python quedonde.py ask "<question>"`.
3. The CLI prints the detected `intent`/`symbol` plus up to 10 summarized rows. Pass `--json` to inspect the raw structured payload (`results` or `details`).
4. When the structural tables are missing, the command emits `[struct] structural tables unavailable` and returns an empty payload, signaling that a re-index is required.

Example commands:

```
python quedonde.py ask "where is update_structural_data defined"
python quedonde.py ask --json "who calls update_structural_data"
python quedonde.py ask "explain update_structural_data"
```

## Fixtures

Sample natural-language prompts and expected intents live in `documentation/fixtures/intent/queries.md`. Each row lists the query, detected intent, and the symbol token that should be extracted. Use this file when adjusting heuristics to avoid regressions.
