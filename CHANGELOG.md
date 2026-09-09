# Changelog

## 0.1.3 — 2026-09-08

- Check every 60 seconds, average up to 15 recent samples, and require three
  consecutive passing checks by default. Keep the minimum warm time at
  2,700 seconds (45 minutes).
- Update CLI help and README examples to match. Timing flags still override
  these defaults. Changing the check interval or sample count clears old
  samples and passing-check counts through the existing state handling.

## 0.1.2 — 2026-09-08

- Show all IDs from Darkbloom's supported-model catalog, current network
  capacity, local scan, and explicit model flags. Refresh the catalog and local
  scan each check using `--config` when provided.
- Mark models absent from the local scan `AUTO-IGNORED; not downloaded or
  filtered out`. Keep their calculations visible and saved. Downloads become
  eligible on the next check; removals revoke pending selections.
- Keep `--model` restrictions separate from table visibility. Show excluded
  rows and raw-score leaders with their explicit or automatic ignore status.
- Retain the last catalog during outages, label stale data, and show unknown
  local availability honestly when scanning fails.
- Check daemon freshness after catalog and pricing reads so a slow request
  cannot make an old snapshot look current.

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
