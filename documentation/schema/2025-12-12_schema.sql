-- Schema snapshot generated UTC date 2025-12-12
-- object: files (table)
CREATE VIRTUAL TABLE files USING fts5(path, content);

-- object: files_config (table)
CREATE TABLE 'files_config'(k PRIMARY KEY, v) WITHOUT ROWID;

-- object: files_content (table)
CREATE TABLE 'files_content'(id INTEGER PRIMARY KEY, c0, c1);

-- object: files_data (table)
CREATE TABLE 'files_data'(id INTEGER PRIMARY KEY, block BLOB);

-- object: files_docsize (table)
CREATE TABLE 'files_docsize'(id INTEGER PRIMARY KEY, sz BLOB);

-- object: files_idx (table)
CREATE TABLE 'files_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID;

-- object: meta (table)
CREATE TABLE meta(path TEXT PRIMARY KEY, mtime REAL);

