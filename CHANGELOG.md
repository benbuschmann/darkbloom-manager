# Changelog

## 0.1.1 — 2026-09-08

- Use minimum warm time and consecutive passing checks in terminal messages,
  with the relevant timing flags beside wait and progress messages. Rename
  `Challenger` to `Candidate` and `Switch ETA` to `Earliest switch`.
- Clarify that the KEEP reason compares models allowed to load. Switching
  rules, defaults, and saved state remain unchanged.

## 0.1.0 — 2026-09-08

First public release.

### Current behavior

- Show exact CLI model IDs in the table and decision lines so they can be
  copied into `--ignore-model`.
- Separate repeated terminal reports with a divider and leave a blank line
  between the model table and highest-score line.
- Discover local downloads through `darkbloom models list --all --json` on
  every check, using the selected provider config.
- Load one model at a time. Ignored models stay visible with their calculations
  but cannot be selected, loaded, or restored from a pending switch.
- Score timestamped average pressure against public output-token prices and
  preference weights. Show unavailable data and ignored raw-score leaders.
- Apply switch costs, margins, consecutive checks, minimum warm time, and
  idle/fresh-daemon checks. Confirm warm-up before starting minimum warm time.
- Save pending state before attempting a launch. Failed, timed-out, or
  interrupted commands cannot trigger repeated launch attempts. Retain command
  errors until fresh daemon state confirms the requested model is warm.
- Respect minimum warm time after a provider restart even when saved state
  contains an older switch time.
- Synchronize the single startup preload before launching the selected model.
- Use stable script, test, and state filenames. Keep the clearer timing flags
  and their older aliases. Copy previous state under the existing process locks.
- Keep the runtime standalone, with offline tests and a README FAQ covering
  setup, API keys, local state, and switching behavior.

### Removed during cleanup

- Paired loading, exploration, and local request/rate observations.
