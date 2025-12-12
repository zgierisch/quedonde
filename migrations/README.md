# Migration Scripts

Store SQLite migration scripts in this directory using zero-padded numeric prefixes, e.g. `001_create_symbols.sql`.

Each file is executed via `sqlite3.Connection.executescript` exactly once and recorded in the `schema_migrations` table. Place SQL statements in the desired order separated by semicolons. Avoid destructive changes; prefer additive migrations and wrap risky operations in `BEGIN`/`COMMIT` blocks.
