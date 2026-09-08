# Contributing

Keep this project a small manager for an already installed Darkbloom provider.
The standalone `warm_model_manager.py` file is the runtime deliverable;
Python's standard library is enough. Changes belong in this repository, which
is authoritative after extraction. Do not maintain parallel edited copies in
other projects.

## Make and test a change

1. Describe the problem and expected behavior in an issue or pull request.
2. Keep changes focused. Add regression tests for selection, state, or launch
   behavior; use fake network/CLI/daemon data and temporary files.
3. Run these commands with Python 3.10 or newer:

   ```sh
   python3 -m py_compile warm_model_manager.py test_warm_model_manager.py
   python3 -m unittest discover -s . -p 'test_*.py' -v
   python3 warm_model_manager.py --version
   python3 warm_model_manager.py --help
   ```

4. Update `CHANGELOG.md` when behavior changes. Update `README.md` whenever
   commands, flags, defaults, setup, or behavior change.

Tests must not require credentials, a real provider, models, or live APIs. Do
not start or restart a real provider as part of automated tests. Any optional
manual live validation must be explicitly authorized by that provider's owner
and reported separately from offline test results.

## Releases

The project is in cleanup toward its first 0.1 public release. Keep
`MANAGER_VERSION = "unreleased"` and add changes under `Unreleased`. Do not
create release tags or publish GitHub releases until the owner explicitly
approves the public release. A cleanup commit or push does not authorize one.

When approved, set the agreed release number, update its test, and add a dated
changelog entry. Filenames and README commands stay independent of release
numbers; state schemas change only when their data format changes. Before
publishing, run the checks, inspect the staged files and outgoing commits, and
verify the destination. Preserve Git history and do not force-push.

Never commit credentials, `.env`, private provider configuration or state,
personal logs, model weights, benchmark output, or provider binaries. Use
synthetic fixtures and placeholders only. Check staged content even when
`.gitignore` appears correct.

## Licensing and upstream changes

Submit original work you have the right to contribute under the MIT license.
Do not copy Darkbloom implementation code or documentation into this project
and label it MIT. Review third-party provenance and required notices before
including any external material. Link to upstream documentation and explain
interfaces in your own words. Call out unresolved license questions before
publishing affected files.

For bug reports, follow the redaction guidance in the README. Never post a real
token or an entire private provider configuration to demonstrate a failure.
