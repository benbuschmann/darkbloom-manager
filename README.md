# Darkbloom warm-model manager

Keeps one downloaded model warm on a running Darkbloom provider. The manager
compares public network pressure and input/output prices, waits for a sustained score
advantage, then switches when the provider is idle.

Shortened example with illustrative values:

```text
Mode:      DRY RUN — no changes enabled
Darkbloom: RUNNING (pid 4242) | warm: qwen3.5-35b-a3b | idle
Discovery: live (local scan; refreshed every check)
Catalog:   live; 7 models; fetched 10:00 AM PDT
Prices:    cached; fetched 9:55 AM PDT
Probes:    OFF (enable with --hourly-probes)
Recovery:  OFF (enable with --recover-routing)
Current:   qwen3.5-35b-a3b warm for 1 hour (since 9:00 AM PDT)

MODEL ID                            NOW  AVG 15m   N   IN$/M  OUT$/M BLEND$/M WEIGHT   SCORE STATUS
---------------------------------------------------------------------------------------------------
* qwen3.5-35b-a3b                 0.683    0.356  15  0.0800  0.7500   0.1805   1.25   0.080
  qwen3.6-35b-a3b-vl-mtp-mxfp8    0.121    0.099  15  0.0500  0.7000   0.1475   1.20   0.018
  gemma-4-26b-qat-4bit            0.843    0.591  15  0.0420  0.2200   0.0687   1.05   0.043
  gpt-oss-20b                     1.004    1.241  15  0.0200  0.1000   0.0320   1.00   0.040
  Qwen3.5-9B                      0.258    0.359  15  0.0800  0.1300   0.0875   1.00   0.031
  gemma-4-26b-8bit                0.000    0.000  15  0.0420  0.2200   0.0687   1.00   0.000 AUTO-IGNORED; not downloaded or filtered out
  EigenLabs/Qwen3.8-27B-4bit-mtp  2.049    2.080  15  0.1500  2.0000   0.4275   1.00   0.889 IGNORED; not downloaded or filtered out

Highest raw score: EigenLabs/Qwen3.8-27B-4bit-mtp [IGNORED] (0.889).
Ranking is before switch cost, required score improvement, consecutive passing checks and minimum warm time.
Ignored and auto-ignored models cannot be loaded.

Decision:  KEEP → qwen3.5-35b-a3b
Switch rule: at least 25% score improvement after switch cost (--switch-improvement-percent).
Reason:    no allowed model scores higher after switch cost (current 0.0803225)
```

[Install](#install) · [Run](#run) · [Routing recovery](#routing-recovery) · [Read the report](#read-the-report) · [FAQ](#faq)

## Install

You need an Apple Silicon Mac with a configured Darkbloom provider, downloaded
models, and Python 3.10–3.14. The script uses Python's standard library and
needs no extra packages. Scoring and switching need no API key; optional
production probes use a token you save separately.

```sh
git clone https://github.com/benbuschmann/darkbloom-manager.git
cd darkbloom-manager
python3 -m unittest discover -s . -p 'test_*.py'
```

Tests use temporary files and fake network, daemon, and launch data. They do not
need Darkbloom installed. CI runs on Linux with each supported Python version
and on macOS with Python 3.14.

Copy `warm_model_manager.py` to run it on another provider Mac, or download the
[latest released script](https://github.com/benbuschmann/darkbloom-manager/releases/latest/download/warm_model_manager.py).
The filename stays the same across releases. Check your copy with
`python3 warm_model_manager.py --version`.

## Run

Preview one check:

```sh
python3 warm_model_manager.py once \
  --ignore-model 'EigenLabs/Qwen3.8-27B-4bit-mtp'
```

For continuous loading:

```sh
python3 warm_model_manager.py run --apply \
  --ignore-model 'EigenLabs/Qwen3.8-27B-4bit-mtp'
```

The defaults check every minute, average up to 15 recent samples, require
three consecutive checks that meet the 25% improvement requirement, and keep a model
warm for at least 45 minutes before replacing it. Use the [switching flags](#switching-rules) to override them.

Add `--hide-ignored` to hide ignored and auto-ignored models from the table
and its ranking. Without it, all models remain visible as before.

Without `--apply`, the manager prints decisions and saves its own state. It
does not launch or stop a provider, edit startup preloads, or send probe prompts. It still reads the catalog
and local model list through Darkbloom, which may migrate an older provider config.

Add `--config /path/to/provider.toml` if your provider uses a custom config.
The catalog, local scan, and live launches all use that path. Use
`--darkbloom /path/to/darkbloom` if the CLI is not on your `PATH`.

## Hourly probes

Add `--hourly-probes` to send a short prompt to the warm model through two
endpoints. This is optional and requires `--apply`.

| Time after enabling | Request |
| --- | --- |
| First check | Local endpoint on this Mac |
| 30 minutes | Production API with `X-Darkbloom-Route: self` |
| 60 minutes | Local endpoint again |
| 90 minutes | Production API again |

This regular schedule gives each endpoint one attempt per hour. The manager reads the warm model
again for every attempt, so the requests follow model changes. Each prompt
asks for `OK`, allows up to 64 output tokens, and has a 30-second socket timeout.
A reasoning model may use the token limit before writing `OK`.

After the manager switches models, it also schedules one production self-route
request **three minutes after the new model is confirmed warm**. The timer starts
at warm-up confirmation, not when the launch command returns. This extra request
uses the same saved production token and `X-Darkbloom-Route: self` header.
It does not move the regular hourly schedule, so it can fall close to a regular
probe. Starting the manager with an already warm model does not add an extra request.

The extra request survives a manager restart and is consumed once, including
on failure or a skip. It is cancelled if the warm model or provider process has
changed. A later manager switch replaces it with a new timer after warm-up.

The local request uses Darkbloom's saved endpoint and local token. The endpoint
record's process ID must match the daemon state, and its address must be loopback.
With probes enabled, future model switches include Darkbloom's `--local-endpoint`
flag. The manager keeps the live endpoint's port, bind address and authentication
setting. If no matching record is available, it uses Darkbloom's authenticated
loopback default on port 8000. It does not restart an already warm provider just
to enable this endpoint. See Darkbloom's
[local endpoint instructions](https://github.com/Layr-Labs/d-inference/blob/master/docs/provider/direct-mode.md).

`HTTP N/A` means no HTTP status was received. For a local `SKIPPED` result, no
request was sent. The reason now distinguishes a missing or unreadable record,
invalid JSON, a missing process ID, and a process mismatch. A mismatch prints
both process IDs; check that `--local-endpoint-file` and `--daemon-state` belong
to the same provider. A missing record can mean the local endpoint is disabled.
Earlier probe builds omitted `--local-endpoint` during model switches, which
could leave the endpoint disabled after a switch.

Save your production API token once on each Mac:

```sh
python3 warm_model_manager.py set-prod-token
```

Paste the token at the hidden prompt. It is saved separately at
`~/.darkbloom/warm-model-manager-prod-token` with owner-only permissions.
Use a key from the account that owns your providers. To replace it, run the
same command again. Tokens are read again for each request.

Then run:

```sh
python3 warm_model_manager.py run --apply --hourly-probes \
  --ignore-model 'EigenLabs/Qwen3.8-27B-4bit-mtp'
```

Production requests go to `https://api.darkbloom.dev/v1/chat/completions`.
Self-route restricts them to your account's providers and does not fall back
to the paid fleet. It can choose another Mac that advertises the same model;
it cannot pin a request to this computer. The local request reaches this Mac.
See Darkbloom's [self-route documentation](https://github.com/Layr-Labs/d-inference/blob/master/docs/provider/self-route.md).

Every report shows the next two probe times, with dates, seconds, timezone and
countdowns. The same two lines appear after each attempt, including skipped
or failed attempts. The display includes the extra request when it is one of
the next two, labeled `production (after switch)`. For example, just after a
model is confirmed warm at 10:10 AM:

```text
Probe 1:   production (after switch) at 2026-09-16 10:13:00 PDT (in 3m)
Probe 2:   production at 2026-09-16 10:30:00 PDT (in 20m)
```

An overdue turn is labeled `overdue`. Its following turn is an estimate until
the first attempt runs or is skipped. Dry runs show that no prompts are
scheduled, even if a live schedule exists in the state file.

Probe diagnostics are always printed when `--hourly-probes` is enabled. A
`SENDING` line identifies the model, endpoint, route, token limit and timeout.
The result line says `SUCCESS`, `FAILED` or `SKIPPED`:

```text
2026-09-16 10:00:00-0700 local probe: SENDING; model="Qwen3.5-9B"; POST http://127.0.0.1:8000/v1/chat/completions; max_tokens=64; timeout=30s
2026-09-16 10:00:02-0700 local probe: SUCCESS; model="Qwen3.5-9B"; HTTP 200; completion received; seconds=2.0; finish_reason=stop; input_tokens=15; output_tokens=1
2026-09-16 10:30:01-0700 production probe: FAILED; model="Qwen3.5-9B"; HTTP 503; HTTP error; seconds=0.4; error=Service Unavailable; model_not_loaded: No owned machine serves this model
```

Success requires a JSON completion containing assistant output. An HTTP 200
with an error or empty output is reported as a failure. Results include elapsed
seconds, the finish reason, token counts and the serving provider ID when
available. `finish_reason=length` means generation ran but reached the output
token limit; it does not mean the model finished its reply.

Failures include the API error code and message, or the connection error.
Non-JSON HTTP errors show a short excerpt. These details are bounded, reduced
to one line, and have credentials redacted. Generated replies and full response
bodies are not printed or saved. Success confirms this inference attempt;
it does not confirm that network assignments have resumed.

An attempt is skipped if the provider is offline, its state is stale, there
isn't exactly one warm model, a switch is pending, or it is serving a request.
Ignored models and models excluded by `--model` are skipped too. Missing tokens
or an unavailable local endpoint skip that endpoint's turn. There are no
immediate retries.

The next endpoint and time survive manager restarts. After downtime, the
manager handles one overdue regular turn and schedules the other 30 minutes later.
It does not replay missed requests. Skipped turns also advance the schedule.
Score-check timing does not change: probes can run between checks. A slow check
or an in-progress switch can delay a probe; a probe that runs past the next
check time delays that check until the request returns. `once --apply --hourly-probes`
handles at most one due request, then exits; use `run` to keep both timers active.

`--prod-token-file PATH` selects another private production token file for
both setup and requests. `--local-endpoint-file PATH` selects another
Darkbloom endpoint record. Its default is `~/.darkbloom/local.json`, or
`$DARKBLOOM_LOCAL_DIR/local.json` when that variable is set. With a custom
`--config`, also select the matching `--daemon-state` and endpoint record.

These probes test inference. We have not established that they restore network
assignments or increase earnings. Turning off `--hourly-probes` stops those
scheduled requests; the saved schedule remains for the next time you enable it.
Checks from `--recover-routing` continue if that separate option is enabled.

## Routing recovery

`--recover-routing` enables one recovery attempt when a warm model answers
locally but production self-routing repeatedly says it is not loaded. It is
off by default. Use `run --apply`; `once` cannot finish the shutdown wait.

Save a production token with `set-prod-token` and use a running provider with
its local endpoint enabled, as described under [hourly probes](#hourly-probes).
The token must belong to the account that owns the provider. Recovery works
with or without `--hourly-probes`:

```sh
python3 warm_model_manager.py run --apply --hourly-probes --recover-routing \
  --ignore-model 'EigenLabs/Qwen3.8-27B-4bit-mtp'
```

The sequence is fixed:

1. Observe the same warm model and provider process for 15 minutes. A model
   change, process change, reported reconnect, or stale state resets that wait.
2. While idle, check local inference and then production self-routing every
   five minutes. Require three consecutive local successes paired with
   **HTTP 503 and error code `model_not_loaded`**. This takes at least ten
   minutes after the first failing check. Zero jobs alone, authentication
   failures, rate limits, timeouts and other server errors do not qualify.
3. After the first qualifying failure, pause score-based switching and the
   regular/after-switch probes while completing the checks. New network
   requests on this provider cancel the shutdown. A failed local check or
   different production result clears the failure count.
4. Once all three checks pass, confirm the provider is idle and its process
   matches Darkbloom's launchd service. Issue `darkbloom stop` once. Wait until
   both the process and service are gone, then count **15 full minutes offline**.
5. Start the same model using the same config and saved local endpoint settings.
   Recheck its local availability and ignore rules first. A changed config,
   removed/excluded model, or external provider start cancels the saved restart.
6. Confirm warm-up, restart the minimum warm-time clock, and run a local/self-route
   check three minutes later. Repeat those checks five minutes apart for up to
   15 minutes after warm-up. If this provider receives network requests, recovery
   is confirmed. Otherwise leave it running and report recovery as unconfirmed.

Recovery has its own checks; it does not move the hourly probe schedule. During
the shutdown, start and verification stages, model switching and other probes
remain paused. Overdue hourly probes resume through their usual one-at-a-time
schedule afterward. Outside recovery, score-based switching keeps its normal rules.

Self-route may reach another provider on your account. A successful HTTP response
alone does not confirm recovery on this Mac. The manager logs the returned provider
ID when supplied and uses an increase in this process's network `requests_served`
counter to confirm local recovery. Local endpoint prompts do not increment that
counter. A successful self-route test reaching this Mac does count; this proves
reachability, not that paid assignments or earnings have resumed. A missing
network counter disables automatic recovery.

There is **one shutdown attempt until this provider receives network traffic or
you explicitly reset recovery**. A manager restart, another model selection, or
an HTTP success from another Mac does not reset that guard. Failed or interrupted
stop/start commands are not retried. The manager checks observed process and
warm-up state because a timed-out command might still have succeeded.

The terminal shows the stage, errors and restart time:

```text
Recovery:  OFFLINE; provider stopped; waiting 15 minutes before starting the same model
Restart:   nvidia-nemotron-3.5-lightning at 2026-09-16 10:45:00 PDT; 12m remaining (after confirmed stop)
Model switching and regular probes paused until recovery finishes.
```

Keep the manager running through the wait. **Ctrl-C during recovery can leave
Darkbloom stopped.** Resume the same command, paths and state file to finish the
saved countdown. Removing `--recover-routing` or `--apply` pauses a saved recovery;
it does not start the provider early. An unreadable recovery state blocks changes.

To clear a completed attempt or cancel checks, stop the manager and run:

```sh
python3 warm_model_manager.py reset-recovery
```

Use the same `--state` path if you changed it. This clears only recovery state and
issues no provider commands. Reset is refused during shutdown, the offline wait
or startup; resume the recovery command first. Changing `--config`, `--daemon-state`,
endpoint/token paths or the CLI path during a saved recovery is also refused.
Normal recovery restarts respect `--config`. Darkbloom's stop command targets
the user's launchd service, so the manager refuses to stop a different process
or a foreground provider.

This is an experimental workaround for the symptoms reported in
[Darkbloom issue #692](https://github.com/Layr-Labs/d-inference/issues/692).
A manual 15-minute shutdown restored traffic on two providers, but the issue
also contains failures after longer gaps. The cause and required waiting time
are unconfirmed. This release's automated validation uses simulated providers;
the manager's recovery procedure has not been tested against a live provider.

## Read the report

In the example, Qwen 3.8 has the highest raw score at `0.889`, but it is ignored.
The manager keeps `qwen3.5-35b-a3b`: its `0.080` score leads the models allowed
to load. A high score alone does not approve a switch.

The `*` marks a currently warm model. Warm means the model is loaded in memory;
`idle` means the provider is not currently serving a request. `Current` shows
how long that model has been warm.

| Table column | Meaning |
| --- | --- |
| `MODEL ID` | Exact ID accepted by `--ignore-model`, including capitalization and namespace. |
| `NOW` | Current public network pressure: active requests divided by warm providers, with a minimum denominator of 1. |
| `AVG 15m` | Average of retained pressure samples. The time label follows your check interval and sample limit. |
| `N` | Number of samples in that average. It can be below the limit after startup or during a data gap. |
| `IN$/M` | Input price per million input tokens. |
| `OUT$/M` | Output price per million output tokens. |
| `BLEND$/M` | Price per million total tokens at the fixed 85% input / 15% output mix, before pressure and weight. |
| `WEIGHT` | Model score multiplier, set with `--weight`. |
| `SCORE` | Average pressure × blended price × weight. |
| `STATUS` | Exclusion or missing-data reason. `IGNORED` is an explicit exclusion; `AUTO-IGNORED` follows local availability or `--model`. |

Pressure describes the network, not the number of requests arriving at your
Mac. Price columns show four decimal places; calculations use the full values.
`N/A` means a value is unavailable, not zero.

By default, `Highest raw score` includes ignored and auto-ignored models and
shows ties. With `--hide-ignored`, it ranks only shown rows. In either view,
the ranking comes before switch cost, required improvement, consecutive passing
checks, and minimum warm time.

| Decision | Meaning |
| --- | --- |
| `KEEP` | Keep the current model. |
| `WOULD SWITCH` | A dry run selected a different model; no launch occurs. |
| `SWITCH` | The manager plans to load a different model. The following log lines report the launch result. |
| `WARMING` | A launch was requested; the manager is waiting for Darkbloom to confirm the model is warm. |
| `DEFERRED` | A model was selected, but a condition such as active work or minimum warm time blocks the launch. |
| `WAIT` | There is no eligible scored target, or the current model's score is unavailable. |

`Reason` explains the decision. `Candidate` shows progress such as
`2/3 consecutive checks passed`, alongside `--switch-after-checks`.
`Earliest switch` is conditional: the candidate must keep passing, the provider
must be online and idle, and minimum warm time must have elapsed. A wait for
warm time names `--min-warm-time`; `Deferred` gives the current blocker.

`Discovery` reports the local scan. `Catalog` reports the supported-model list.
`Prices` shows whether prices were fetched, cached, or retained after a failed
refresh, with their fetch time.

## How scores work

```text
pressure = active requests / max(1, warm providers)
blended price = (input price × 0.85) + (output price × 0.15)
score = average pressure × blended price × weight
```

Prices are USD per million tokens. Every model uses the same fixed mix:
85% input tokens and 15% output tokens. The mix is built into the script.
Prices still refresh from Darkbloom's public endpoint.

For example, an input price of `$0.08/M` and an output price of `$0.13/M`
give a blended price of `$0.0875/M` total tokens. At average pressure `2.0`
and weight `1.00`, the score is `0.175`.

The score compares models at that assumed mix. It does not measure your
provider's throughput, actual token mix, or payouts, and does not reproduce
per-request billing rounding.

Samples expire after `check-every × average-samples` seconds, including while
the manager is stopped. With the defaults of `60` and `15`, the average contains
at most 15 samples from the past 15 minutes. The table's `N` column shows how many are
available. Changing either setting clears the old samples and resets the
number of consecutive passing checks.

| Model ID | Default weight |
| --- | ---: |
| `qwen3.5-35b-a3b` | 1.25 |
| `qwen3.6-35b-a3b-vl-mtp-mxfp8` | 1.20 |
| `gemma-4-26b-qat-4bit` | 1.05 |
| `gpt-oss-20b` | 1.00 |
| Other models | 1.00 |

Set a weight with `--weight MODEL_ID=WEIGHT`. Repeat it for more models.
For example, `--weight Qwen3.5-9B=1.25` increases that model's score by 25%.
Weights must be positive; they can also name ignored models or future downloads.

## Switching rules

| Flag | Default | What it controls |
| --- | --- | --- |
| `--check-every SECONDS` | 60 | Seconds between checks. Minimum: 60. |
| `--average-samples COUNT` | 15 | Maximum number of recent pressure samples to average. |
| `--switch-after-checks COUNT` | 3 | Consecutive checks the same candidate must pass before switching. |
| `--min-warm-time SECONDS` | 2700 | Minimum time to keep a warm model before replacing it. |

The old names still work as aliases: `--interval`, `--history`,
`--confirmations`, and `--min-dwell`, respectively.

Before comparing a candidate with the current model, the manager discounts its
score for the estimated time lost while switching:

```text
adjusted candidate score = candidate score × (decision-horizon − switch-cost) / decision-horizon
required score = current score × (1 + switch-improvement-percent / 100)
```

The default requires at least 25% improvement after that discount. For a current
score of `0.010`, an adjusted candidate score of `0.013` is a 30% improvement and
passes. An adjusted score of `0.012` is a 20% improvement and fails. There is no
fixed score increase to clear.

Use `--switch-improvement-percent 25` for 25%, or
`--switch-improvement-percent 1` for 1%. A value of `0.25` means 0.25% with this
flag. The report shows the requirement and the candidate's improvement after
switch cost. A failing check resets the consecutive passing count.

Equal scores keep the current model, including when both scores are zero.
A positive candidate can pass against a zero current score; it still needs
the required consecutive checks and minimum warm time.

| Flag | Default | What it controls |
| --- | --- | --- |
| `--switch-improvement-percent PERCENT` | 25 | Required percentage improvement after switch cost. Must be nonnegative. |
| `--switch-cost` | 300 seconds | Estimated time lost while changing models. |
| `--decision-horizon` | 3600 seconds | Period used to weigh that lost time. Must exceed switch cost. |
| `--warmup-timeout` | 180 seconds | Time to wait for the requested model to become warm before reporting loading as overdue. |
| `--pricing-refresh` | 900 seconds | Time between price-cache refreshes. |

The older `--relative-margin 0.25` form still means 25%. Use either that flag
or `--switch-improvement-percent`, never both. `--absolute-margin` has been
removed; the manager reports an error if it appears in your command.

Changing the percentage, switch cost, or decision horizon resets live and
dry-run passing-check counts. Pressure history and minimum warm time remain.

If no eligible model is currently warm, the manager selects the highest scored
eligible model without waiting for consecutive passing checks. It still checks
daemon health, active requests, and pending warm-up. An empty warm list has no minimum
warm time to preserve; an existing warm selection does.

A live switch runs `darkbloom start` with one `--model` and
`--idle-timeout 0`. That restarts the provider. Before launching, the manager
sets `backend.preload_models` to a one-element list containing the selected
model, preserving the rest of the config text. It waits until fresh daemon
state confirms exactly that model is warm before starting the minimum warm
time. A provider restart also starts a new minimum warm period, even if the
manager's saved switch time is older.

## Discovery and ignored models

The table combines three sources on every check:

| Source | What it supplies |
| --- | --- |
| `darkbloom models catalog --json` | Supported model IDs from the coordinator in your provider config, including models you have not downloaded. |
| Public network capacity | Model IDs and current pressure, including IDs missing from the catalog response. |
| `darkbloom models list --all --json` | Locally discovered models. Only models in a successful local scan can be loaded. |

Both Darkbloom commands respect `--config`. The catalog uses concrete model
IDs, rather than the consumer-facing aliases shown in some Darkbloom pages.
Explicit `--model` and `--ignore-model` IDs also remain visible.

Models absent from the local scan show
`AUTO-IGNORED; not downloaded or filtered out`. Their pressure, price, weight,
and score are still calculated and saved when network data is available.
They cannot be selected or loaded. Once a download appears in the local scan,
that automatic exclusion clears. Removing a download excludes it again and
revokes any saved pending selection for it. The manager does not download models.

Repeated `--model MODEL_ID` flags restrict loading candidates and set their
tie order. Other rows remain visible with `AUTO-IGNORED; outside --model selection`.
Without that restriction, the four weighted models above retain their tie
order, followed by other IDs alphabetically. Explicit ignore IDs appear last
unless placed earlier by `--model`. Each launch still requests one model.

Use `--ignore-model MODEL_ID [MODEL_ID ...]`, or its `--ignore` alias, to exclude
models from loading. Give it space-separated IDs, repeat the flag, or combine
both forms. IDs are exact and case-sensitive. For example:

```sh
python3 warm_model_manager.py run --apply \
  --ignore-model 'EigenLabs/Qwen3.8-27B-4bit-mtp' 'Qwen3.5-9B' \
  --hide-ignored
```

Put `run` or `once` before the flags. Each ignore flag takes one or more IDs;
the next flag ends that list. Repeated single-model flags still work.

Ignored models stay in the table by default, even when absent from the local list. Their
rows show `IGNORED` alongside pressure, average, both prices, blend, weight, and score.
Their calculations are saved too. An explicit ignore stays in effect even if
you download the model later.

With `--hide-ignored`, both `IGNORED` and `AUTO-IGNORED` rows are hidden.
The report shows how many models are displayed and how many are hidden in each
group. This also hides models excluded by `--model`. The highest-score line
is labeled `Highest raw score (shown models)` and ranks only the displayed rows.
If all rows are hidden, the table says `No models to show.`

Hiding rows changes only the display. Their timestamped samples, prices, scores,
and exclusion reasons remain in the state file. Toggling the flag preserves
passing-check counts and warm-up tracking. The provider-status line still
reports what is actually warm, even if that model's table row is hidden.

Darkbloom's `--all` bypasses the enabled-model config filter, but its scanner can
still omit downloads that exceed available memory. The catalog's JSON output
does not report download status. The manager therefore cannot distinguish a
missing download from a filtered download; the status says both. Listing also
skips the serving command's runtime-capability filter. A listed model can still
fail to run on your GPU. Use an explicit ignore for models your provider cannot run.

This behavior was checked on September 8, 2026 against the upstream
[list command](https://github.com/Layr-Labs/d-inference/blob/efcde6334ddf95a98e7c5353329abc52e195e9f2/provider-swift/Sources/darkbloom/ModelsCommand.swift),
[catalog client](https://github.com/Layr-Labs/d-inference/blob/efcde6334ddf95a98e7c5353329abc52e195e9f2/provider-swift/Sources/ProviderCore/Models/ModelCatalogClient.swift),
[runtime filtering](https://github.com/Layr-Labs/d-inference/blob/efcde6334ddf95a98e7c5353329abc52e195e9f2/provider-swift/Sources/darkbloom/Darkbloom.swift),
and [model scanner](https://github.com/Layr-Labs/d-inference/blob/efcde6334ddf95a98e7c5353329abc52e195e9f2/provider-swift/Sources/ProviderCore/Models/ModelScanner%2BDiscovery.swift).

## FAQ

### Do I need a Darkbloom API key?

Scoring and model switching need no API key. Optional
[hourly probes](#hourly-probes) and [routing recovery](#routing-recovery) use Darkbloom's saved local token and a
production API token you save with `set-prod-token`.

The manager reads public capacity and pricing endpoints and uses your installed
Darkbloom CLI and provider config for local operations. It does not read a
`.env` file. Darkbloom itself must already be configured and
running; follow its [provider setup instructions](https://github.com/Layr-Labs/d-inference/blob/master/docs/provider/installation.md).

### What is saved locally?

The manager writes `~/.darkbloom/warm-model-manager-state.json` after each check,
including dry runs. The file is replaced atomically and has owner-only read and
write permissions.

| Saved data | Purpose |
| --- | --- |
| Timestamped pressure samples | Rebuild each model's rolling average after a restart. |
| Latest score snapshot | Retain pressure, average, input/output prices, blended price, token mix, weight, score, local availability, and explicit or automatic ignore reasons. |
| Price cache and fetch time | Reuse both prices and public fallback prices between refreshes and identify stale prices. |
| Discovered model IDs, scan times, and scan errors | Show the last inventory when discovery fails. |
| Catalog IDs, source, fetch time, and errors | Keep catalog rows visible during an outage and identify stale catalog data. |
| Current model, process identity, warm-start time, and last switch time | Track how long the model has been warm. |
| Candidate and count of consecutive passing checks | Resume progress toward `--switch-after-checks` across restarts. Live and dry-run counts are separate. |
| Pending target, launch time, and any command error | Wait for warm-up without issuing repeated restarts. |
| Percentage requirement, switch cost, and decision horizon | Reset passing-check counts when the switch rule changes. |
| Last decision, scoring mix, selection policy, timing settings, and format/release metadata | Explain the last check and detect incompatible saved settings. |
| Probe schedule and latest result | Remember the next endpoint and time, plus the last attempt's model, outcome, duration, HTTP status, redacted error or skip reason, finish reason, token counts and serving provider ID when available. |
| Extra probe after a switch | Keep the target, provider process identity, warm-up confirmation and due time until the one-shot request is attempted or skipped. Retain its latest result separately from the hourly probe result. |
| Optional routing recovery | Stage, model/process identity, one network counter baseline, reconnect count, failure count, next check, stop/start intent, offline deadline, elapsed-time reference, config digest and paths, local endpoint launch flags, latest redacted check results and the one-attempt guard. No config contents or tokens. |

Each save replaces the latest snapshot and keeps a bounded sample window.
The state file contains no prompts, generated replies or API keys. Probe errors
retain only the short, redacted description from the result line. The production
token lives in its separate private file; the local token stays in Darkbloom's
endpoint record. Probe logs go to the terminal. The manager does not collect
request histories or per-model request rates. With recovery enabled, it retains
one network request counter baseline to detect traffic on this provider. Catalog, discovery, and launch errors
can contain local paths or CLI error text.

Use `--state /path/to/state.json` for another location. On the first run with
the new default path, the manager copies the previous default state if present
and leaves the original file in place. It holds the older managers' locks
before copying. An existing new state file wins; custom state paths are not
migrated. A release-number change does not reset state.

The saved ignore list records the last run's settings. Keep `--ignore-model`
in your launch command; saved settings do not replace command-line options.

### Can I delete the state file?

Do not delete state during routing recovery: it holds the restart deadline and
the guard against repeated shutdowns. Use `reset-recovery` for a completed attempt.
Stop the manager first. Deleting state removes rolling samples, progress toward
the required consecutive checks, and pending warm-up tracking. For a deliberate
fresh start, use a new `--state` path after stopping the old process. Deleting the default file
can cause the retained previous state to be imported again.

### What happens when data is unavailable?

Missing pressure or price appears as `N/A`. Recent averages remain visible
until they expire, but missing current pressure prevents a fresh score. A price
refresh failure can use the last cache containing both prices, labeled with
its fetch time. Models absent from the pricing response use Darkbloom's public
fallback input and output prices when available.

A listed model missing either price shows `N/A` for that component, the blend,
and the score. Its known price stays visible. The manager does not replace a
missing component with zero or a fallback. A zero price is accepted only when
explicitly supplied; the blended price must be positive to allow selection.
If the current model's score is unavailable, switching waits for usable data.

If the catalog request fails, the last catalog remains visible as `stale cache`.
Without a cached catalog, the report says `Catalog: unavailable` and still shows
IDs from the capacity feed, local scan, and your flags. Catalog membership alone
never permits a launch.

If local discovery fails, the last inventory remains visible and new launches
are blocked. Local availability is unknown, and the rows say `local scan unavailable`.
With `--hide-ignored`, those rows are hidden until a successful scan confirms
local availability. The `Discovery: unavailable` message remains visible.

### Why hasn't it switched?

Read `Reason`, `Candidate`, and `Deferred` in the report. The candidate may need
more passing checks (`--switch-after-checks`), the current model may need more
warm time (`--min-warm-time`), or the improvement after switch cost may be below
`--switch-improvement-percent`. A busy, stopped, or stale provider also blocks
a switch. Dry runs use the same rules.

A failed or timed-out launch command, load error, or overdue warm-up blocks
automatic retries. The pending target is saved before attempting the launch;
fresh daemon state can still confirm success if the command timed out. Inspect
`darkbloom status` and the provider logs, fix the loading problem, then stop the
manager before editing pending state.

### What does Ctrl-C stop?

It stops the manager after any in-flight check or command finishes. The
provider keeps its last model, startup preload, and always-warm idle setting.
The manager runs in the foreground and installs no background service.
If routing recovery has stopped Darkbloom, Ctrl-C leaves it stopped. Resume the
same recovery command to finish the saved wait and startup.

### How do I update an older installation?

Stop the manager first. In a Git checkout, run `git pull` to get the latest
source. For a standalone copy, run this in the folder containing your script
to replace it with the latest public release:

```sh
curl -fL https://github.com/benbuschmann/darkbloom-manager/releases/latest/download/warm_model_manager.py \
  -o warm_model_manager.py.new &&
chmod +x warm_model_manager.py.new &&
mv warm_model_manager.py.new warm_model_manager.py &&
python3 warm_model_manager.py --version
```

Remove obsolete flags as described below, then run your launch command.
Replacing the script leaves saved state intact.
[Release notes and checksums](https://github.com/benbuschmann/darkbloom-manager/releases/latest)
are available on GitHub. Script and state filenames stay the same across releases.

`--hide-ignored` is optional. Existing commands keep the full table. To hide
excluded models, add the flag; your saved state and switching rules stay intact.

The percentage-based switch rule replaces the old fixed `0.01` score requirement.
Remove `--absolute-margin` if you used it. The default remains 25% improvement;
`--relative-margin 0.25` still works, or use `--switch-improvement-percent 25`.
The first check after this upgrade resets live and dry-run passing-check counts
once. Pressure history, minimum warm time, and any already-issued pending launch
stay in place.

When upgrading from output-only scoring, the manager fetches both prices before
scoring. If that fetch fails, prices remain unavailable until it succeeds.
The formula change resets live and dry-run passing-check counts once. Retained
pressure samples, minimum warm time, and any already-issued pending launch stay
in place; normal sample expiration and timing-setting resets still apply.

Remove old pairing and exploration flags, including `--no-pair` and
`--no-exploration`. Those features are gone. Pending selections that contain
multiple models are discarded. Only one model can satisfy warm-up.

Run one manager per provider. The lock prevents overlapping old and new
processes; do not delete a lock file while a manager is running.

## Files and support

Darkbloom's daemon state defaults to `~/.darkbloom/daemon-state.json`. Override
it with `--daemon-state` or `DARKBLOOM_STATE_FILE`. The provider config defaults
to `~/.config/darkbloom/provider.toml`.

Report bugs in [GitHub issues](https://github.com/benbuschmann/darkbloom-manager/issues)
with the manager, Darkbloom, Python, and macOS versions, your chip and RAM, model
IDs, a redacted command, and the relevant report lines. Leave out tokens and
private config files. Development checks are in [CONTRIBUTING.md](CONTRIBUTING.md).

This project's code and documentation use the [MIT license](LICENSE).
Darkbloom is installed separately and has its own
[license](https://github.com/Layr-Labs/d-inference/blob/master/LICENSE).
