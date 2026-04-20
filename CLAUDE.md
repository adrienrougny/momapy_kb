# Repo conventions for Claude

## Layout

- Package code: `src/momapy_kb/`
- Tests: `tests/` — pytest, with backend-parametrized fixtures in `conftest.py`
  (`neo4j`, `falkordb`, `falkordblite`) driven by testcontainers
- Fixture map files: `tests/sbgn/maps/**/*.sbgn`,
  `tests/celldesigner/maps/*.xml` — parametrize file-based tests by globbing
  these directories
- Implementation plans: `plans/*.md`
- User docs: `docs/` (mkdocs)

## Tooling

- Dependency manager: `uv` (lock file is `uv.lock`)
- To pick up a newer version of a single dep:
  `uv lock --upgrade-package <name> && uv sync`
- Build backend: hatchling + hatch-vcs (version from git tags matching
  `^v(?P<version>([0-9].*))`)
- Test runner: `pytest` via `tox` (envs: `py{310,311,312,313}-{min,all}`)

## Dependency-bump policy

When a dependency ships a bug fix that affects this package, bump the minimum
in `pyproject.toml` (e.g. `"momapy>=0.8.0"`). A lock-file update alone isn't
enough — downstream installers may otherwise resolve to the broken version.

## Built on

- `fieldz-kb` — auto-serializes dataclass-like objects to LPG nodes/edges by
  walking their fields. momapy-kb extends this for momapy types.
- `momapy` — the map/model/layout dataclasses being serialized.
- `pylpg` — the labeled-property-graph session abstraction.

## Save flow (important for anything that emits edges)

All save entry points on `momapy_kb.lpg.session.Session` funnel through
`save_from_objects`, which delegates to
`fieldz_kb.lpg.session.Session.save_from_objects`. The `integration_mode`
parameter (`"id"` or `"hash"`) controls node dedup:

- `"id"`: keys in the internal `object_to_node` dict are `id(obj)`
- `"hash"`: keys are the objects themselves (momapy dataclasses are frozen)

Any post-pass that needs to look up a node from a momapy element must respect
this keying. See `plans/membership_edges.md` for an example.

## Testing conventions

- Tests for a module `momapy_kb.<mod>` live in `tests/test_<mod>.py` (mirror
  the package structure; don't mix concerns in one file)
- File-based tests parametrize over real fixture maps rather than hand-crafted
  inputs
- No `if __name__ == "__main__": pytest.main(...)` boilerplate — run with
  `pytest` or `tox`
