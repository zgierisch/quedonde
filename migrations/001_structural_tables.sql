-- Phase 1 structural storage primitives
BEGIN;

CREATE TABLE IF NOT EXISTS symbols (
    path TEXT NOT NULL,
    symbol TEXT NOT NULL,
    kind TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    PRIMARY KEY (path, symbol, kind)
);
CREATE INDEX IF NOT EXISTS idx_symbols_symbol ON symbols(symbol);
CREATE INDEX IF NOT EXISTS idx_symbols_path ON symbols(path);

CREATE TABLE IF NOT EXISTS edges (
    src_path TEXT NOT NULL,
    src_symbol TEXT NOT NULL,
    relation TEXT NOT NULL,
    dst_path TEXT,
    dst_symbol TEXT,
    CHECK (relation IN ('calls', 'imports', 'includes', 'references', 'owns'))
);
CREATE INDEX IF NOT EXISTS idx_edges_src_symbol ON edges(src_symbol);
CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);
CREATE INDEX IF NOT EXISTS idx_edges_dst_symbol ON edges(dst_symbol);

CREATE TABLE IF NOT EXISTS annotations (
    path TEXT NOT NULL,
    symbol TEXT,
    tag TEXT NOT NULL,
    line INTEGER,
    CHECK (tag IN ('legacy', 'bridge', 'orchestrator', 'deprecated', 'temporary'))
);
CREATE INDEX IF NOT EXISTS idx_annotations_tag ON annotations(tag);
CREATE INDEX IF NOT EXISTS idx_annotations_path ON annotations(path);

COMMIT;
