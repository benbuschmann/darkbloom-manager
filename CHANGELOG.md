# Changelog

## 0.1.10 — 2026-09-16

- Put `now`, `next`, `score` and `sources` in a left gutter. Show score checks,
  conditional switch times, recovery timers and probes in one ordered timeline.
- Show both hourly endpoints alongside a pending post-switch self-route request.
  Refresh the timeline after each probe attempt; keep HTTP results and errors.
- Replace the default price table with a score ladder: raw score, relative bar,
  cost-adjusted percentage versus the warm model, average pressure, blended
  price and weight. Keep ignored rows and missing-data labels visible by default.
- Add `--columns full` for the detailed table. Keep switching, recovery, saved
  samples and timers unchanged. Use plain output for redirected logs and honor
  `NO_COLOR` in terminals.
- Update README examples and add offline coverage for timer ordering, paused
  recovery, conditional switch times, score comparisons and both displays.

## 0.1.9 — 2026-09-16

- Add optional `--recover-routing` for continuous live runs. Observe a model
  warm for 15 minutes, then require three local successes paired with production
  self-route `503 model_not_loaded` responses, five minutes apart.
- Pause switching and other probes while confirming the failure. Cancel the
  shutdown if this provider receives network requests. Ignore timeouts, key
  errors, rate limits and ordinary idle periods as recovery triggers.
- Verify the matching launchd process, stop once, and wait 15 minutes from
  confirmed shutdown before starting the same model. Preserve config selection,
  local endpoint settings, ignore exclusions and single-model preloads.
- Persist the recovery stage, offline deadline and command intents. Resume
  across manager restarts without repeating commands or shortening the wait.
  Refuse changed paths/config, excluded models and conflicting external starts.
- Check routing after warm-up and confirm recovery only through this provider's
  network request counter. An account-wide self-route success alone is not proof.
  Keep the one-attempt guard until local network traffic returns or the operator
  uses `reset-recovery`; do not retry failed recovery in a shutdown loop.
- Show recovery stages, restart countdowns and redacted probe results. Document
  setup, saved state, Ctrl-C behavior and the limits of this experimental workaround.
- Add offline recovery, interruption, exclusion and process-identity tests.
  No live provider was stopped, started or probed during development.

## 0.1.8 — 2026-09-16

- With `--hourly-probes`, send one extra production self-route prompt three
  minutes after a manager-selected model is confirmed warm. Keep its timer
  separate from the hourly cadence, show it in the next-two-probes display,
  and consume it before sending so interruptions cannot replay it.
- Keep the local endpoint enabled during model switches when probes are on.
  Preserve a live endpoint's port, bind address and authentication setting;
  otherwise use Darkbloom's authenticated loopback default. Do not restart a
  provider just to repair an endpoint.
- Replace the combined local-endpoint skip message with specific missing-file,
  unreadable-file, invalid-JSON and process-ID errors. Show both process IDs
  when the endpoint record belongs to another process.
- Add optional `--hourly-probes`: alternate a local prompt and a production
  self-route prompt 30 minutes apart, once per hour for each endpoint.
  Read the currently warm model each time and require `--apply` to send.
- Save the next probe before sending. Resume after restarts without replaying
  missed requests. Skip busy, offline, stale, switching or excluded models.
- Add `set-prod-token` for hidden token entry into a separate owner-only file,
  plus path overrides for that token and Darkbloom's local endpoint record.
- Show the next two probe times and countdowns in every report and after every
  attempt. Label overdue turns, estimated follow-up times and disabled probes.
- Print sending and result diagnostics with success/failure/skip status,
  elapsed time, finish reason, token counts and redacted API or connection errors.
  Check completion bodies so HTTP 200 alone is not reported as success.
- Log probe times, models, HTTP results and returned provider IDs. Keep tokens,
  generated replies and full response bodies out of logs and state. Reject
  redirects and require the local endpoint to match the running provider
  process at a loopback address.

## 0.1.7 — 2026-09-11

- Add `--hide-ignored` to hide ignored and auto-ignored models from the table
  and its ranking. Keep the full table by default. Show displayed and hidden
  counts when the flag is used, and handle an empty table without a false leader.
- Accept multiple space-separated model IDs after each `--ignore-model` or
  `--ignore` flag. Repeated flags still work in either display mode.
- Keep hidden models' calculations and timestamped samples in saved state.
  Changing display mode does not reset passing checks or warm-up tracking.
  Ignored models remain blocked from selection and saved pending launches.
- Update help, README examples, and tests for both display modes, grouped
  ignores, discovery changes, and missing data.

## 0.1.6 — 2026-09-09

- Replace the fixed `0.01` score requirement with one percentage-based switch
  rule. `--switch-improvement-percent 25` requires at least 25% improvement
  after switch cost, regardless of score size. The default is 25%.
- Show the percentage requirement and the candidate's improvement in reports.
  Keep equal scores on the current model, including zero-score ties.
- Keep `--relative-margin 0.25` as the older form of 25%. Reject
  `--absolute-margin` with instructions to use the percentage flag.
- Reset live and dry-run passing-check counts when adopting the new rule or
  changing the percentage, switch cost, or decision horizon. Preserve pressure
  history, minimum warm time, and already-issued pending launches.
- Update README examples, flag descriptions, and upgrade notes. Add tests for
  small scores, percentage boundaries, switch costs, zero scores, and state upgrades.

## 0.1.5 — 2026-09-09

- Rename the table's `PREF` column to `WEIGHT`, matching `--weight`. Use
  weight consistently in code, help text, formulas, tests, and README examples.
- Save each model's score multiplier under `weight` in the latest score
  snapshot. Existing snapshots are replaced on the next check; pressure
  history, passing-check counts, and pending warm-up remain intact.
- Keep the score formula, weights, timing defaults, and switching rules unchanged.

## 0.1.4 — 2026-09-09

- Score every model using average pressure × (85% input price + 15% output
  price) × weight. Keep existing weights, timing defaults, switch costs,
  and relative/absolute score requirements.
- Show `IN$/M`, `OUT$/M`, and `BLEND$/M` with four decimal places. Replace
  `PROJ$/M` and save both prices, the blend, and the fixed token mix with each
  score snapshot, including ignored and auto-ignored models.
- Cache both prices and their public fallbacks. Missing or invalid components
  stay unavailable and cannot produce a score; an explicit zero component is
  allowed when the blend remains positive.
- Refresh output-only caches and reset passing-check counts once when adopting
  the new formula. Preserve timestamped pressure history, minimum warm time,
  and already-issued pending launches.
- Update README formulas, table explanations, examples, and upgrade notes.

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
  model weights. Show unavailable data and ignored raw-score leaders.
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
