# Extension Classifier Matrix

This table defines the deterministic mapping used by `quedonde.py` when deciding which structural extractor to apply during indexing. Only the listed extensions participate in structural symbol extraction; all other files remain lexical-only.

| Language group | Extensions | Symbol kinds emitted |
| --- | --- | --- |
| python | `.py` | `class`, `function`, `method`
| cpp | `.c`, `.cc`, `.cpp`, `.cxx`, `.h`, `.hpp`, `.hh` | `class`, `namespace`, `function`
| javascript | `.js`, `.jsx` | `class`, `function`
| typescript | `.ts`, `.tsx` | `class`, `function`
| json | `.json` | `key` (top-level keys)
| yaml | `.yml`, `.yaml` | `key` (top-level keys)
| markdown | `.md` | `heading`

Each classifier bucket points to a dedicated regex/line-heuristic extractor implemented in `quedonde.py` under the "Structural extraction" section. If an extension is not present in this matrix, the structural tables will contain no rows for that file even when it is indexed.
