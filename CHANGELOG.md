# Changelog

## Unreleased

Cleanup toward the first 0.1 public release. A release date has not been set.

### Current behavior

- Discover local downloads through `darkbloom models list --all --json` on
  every check, using the selected provider config.
- Load one model at a time. Ignored models stay visible with their calculations
  but cannot be selected, loaded, or restored from a pending switch.
- Score timestamped average pressure against public output-token prices and
  preference weights. Show unavailable data and ignored raw-score leaders.
- Apply switch costs, margins, consecutive checks, minimum warm time, and
  idle/fresh-daemon checks. Confirm warm-up before starting minimum warm time.
- Synchronize the single startup preload before launching the selected model.
- Use stable script, test, and state filenames. Keep the clearer timing flags
  and their older aliases. Copy previous state under the existing process locks.
- Keep the runtime standalone, with offline tests and a README FAQ covering
  setup, API keys, local state, and switching behavior.

### Removed during cleanup

- Paired loading, exploration, and local request/rate observations.
- Numbered development-release headings and release links from these notes.
  The runtime identifies itself as `unreleased` until a public release is approved.
