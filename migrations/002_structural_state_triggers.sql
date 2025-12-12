BEGIN;

CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT
);

INSERT OR REPLACE INTO state(key, value)
VALUES('structural_version', strftime('%s','now'));

CREATE TRIGGER IF NOT EXISTS trg_symbols_touch_insert
AFTER INSERT ON symbols
BEGIN
    INSERT OR REPLACE INTO state(key, value)
    VALUES('structural_version', strftime('%s','now'));
END;

CREATE TRIGGER IF NOT EXISTS trg_symbols_touch_update
AFTER UPDATE ON symbols
BEGIN
    INSERT OR REPLACE INTO state(key, value)
    VALUES('structural_version', strftime('%s','now'));
END;

CREATE TRIGGER IF NOT EXISTS trg_symbols_touch_delete
AFTER DELETE ON symbols
BEGIN
    INSERT OR REPLACE INTO state(key, value)
    VALUES('structural_version', strftime('%s','now'));
END;

CREATE TRIGGER IF NOT EXISTS trg_edges_touch_insert
AFTER INSERT ON edges
BEGIN
    INSERT OR REPLACE INTO state(key, value)
    VALUES('structural_version', strftime('%s','now'));
END;

CREATE TRIGGER IF NOT EXISTS trg_edges_touch_update
AFTER UPDATE ON edges
BEGIN
    INSERT OR REPLACE INTO state(key, value)
    VALUES('structural_version', strftime('%s','now'));
END;

CREATE TRIGGER IF NOT EXISTS trg_edges_touch_delete
AFTER DELETE ON edges
BEGIN
    INSERT OR REPLACE INTO state(key, value)
    VALUES('structural_version', strftime('%s','now'));
END;

CREATE TRIGGER IF NOT EXISTS trg_annotations_touch_insert
AFTER INSERT ON annotations
BEGIN
    INSERT OR REPLACE INTO state(key, value)
    VALUES('structural_version', strftime('%s','now'));
END;

CREATE TRIGGER IF NOT EXISTS trg_annotations_touch_update
AFTER UPDATE ON annotations
BEGIN
    INSERT OR REPLACE INTO state(key, value)
    VALUES('structural_version', strftime('%s','now'));
END;

CREATE TRIGGER IF NOT EXISTS trg_annotations_touch_delete
AFTER DELETE ON annotations
BEGIN
    INSERT OR REPLACE INTO state(key, value)
    VALUES('structural_version', strftime('%s','now'));
END;

COMMIT;
