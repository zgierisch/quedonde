# Changelog

## [Unreleased]

### Added
- `--lines` flag to include matching line numbers for content searches.
- `--title` flag to filter results by path segments while keeping content searches intact.
- Context collection now reuses a single pass to supply both snippets and line numbers, enabling richer `--context` output.
- README guidance covering the interaction between `--paths`, `--title`, and other flags.

### Fixed
- Normalize content queries containing punctuation so SQLite FTS no longer throws syntax errors.
- Handle trailing parentheses in content queries without triggering FTS syntax errors.
