# Project instructions

This repository is the authoritative source for the portable Darkbloom
warm-model manager. Keep `warm_model_manager.py` standalone, portable to
supported provider Macs, and limited to the Python standard library plus the
installed Darkbloom CLI. Avoid frameworks, extra services, and dependency-heavy
release tooling. Do not maintain divergent copies in a parent workspace.

- Keep runtime, test, and default state filenames free of release numbers.
  Keep state schema identifiers separate from `MANAGER_VERSION`.
- This is cleanup work toward the first 0.1 public release. Keep
  `MANAGER_VERSION = "unreleased"` and record changes under `Unreleased`.
- Do not create release tags, publish GitHub releases, or assign a release
  number until the owner explicitly says the public release is ready. Requests
  to edit, test, commit, or push cleanup work are not release authorization.
- Update `CHANGELOG.md` when behavior changes. Add a dated release entry only
  when a release is explicitly authorized; use the actual release date.
- Update `README.md` for any changed command, flag, default, setup, or behavior.
- Test relevant behavior before publishing. Run syntax checks and the offline
  unittest suite described in `CONTRIBUTING.md`.
- Mock network calls, provider commands, and daemon state in tests. Use temporary
  files. Never start or restart a real provider without explicit owner approval.
- Preserve timestamp aging, confirmation/dwell and warm-up safeguards, ignore
  exclusions across every selection path, and preload synchronization.
- Keep loading strictly single-model: each launch and startup preload must name
  exactly one selected model; do not reintroduce companion loads.
- Never commit secrets, `.env`, personal provider data, private configuration,
  raw logs, downloaded weights, or Darkbloom binaries. Inspect exact staged
  files and outgoing history before publishing.
- Preserve upstream history and unrelated work. Do not force-push.
- Review third-party provenance; MIT covers this project's original material,
  not Darkbloom's separately licensed code or documentation.

When reporting a change, state what changed, the version, tests run, and any
limits of the validation. Distinguish offline tests from live provider checks.
