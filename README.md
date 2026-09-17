# Darkbloom warm-model manager

Keeps one downloaded model warm on a running Darkbloom provider. The manager
compares public network pressure and input/output prices, waits for a sustained score
advantage, then switches when the provider is idle.

Example with illustrative values, with hourly probes and routing recovery enabled:

```text
================================================================================================================
Darkbloom warm model manager 0.1.14    2026-09-16 07:31:52 PDT

now      nvidia-nemotron-3.5-lightning  warm 1 hour 10 minutes  serving a request    LIVE, KEEP
         session: 168 requests | 124,521 tokens
         gemma-4-26b-qat-4bit: +261.17% improvement after switch cost; 2/3 consecutive checks passed

next     07:32:52  check 3 of 3 for gemma-4-26b-qat-4bit; earliest switch if checks still pass and the provider is online and idle
         07:36:00  recovery check, local then production self-route
         08:00:00  probe, local endpoint
         08:30:00  probe, production self-route
         +3m after a switch is confirmed warm: one production self-route request

score    MODEL ID                          SCORE                  VS WARM        AVG   BLEND$/M  WEIGHT
           gemma-4-26b-qat-4bit            0.032  ████████████      +261%  avg 0.445   $0.0687  ×1.05
           qwen3.5-35b-a3b                 0.023  ████████          +154%  avg 0.100   $0.1805  ×1.25
           qwen3.6-35b-a3b-vl-mtp-mxfp8    0.015  ██████             +71%  avg 0.086   $0.1475  ×1.20
           Qwen3.5-9B                      0.010  ████               +14%  avg 0.116   $0.0875  ×1.00
         * nvidia-nemotron-3.5-lightning   0.008  ███                warm  avg 0.099   $0.0823  ×1.00

         Highest raw score: gemma-4-26b-qat-4bit (0.032).
         score = average pressure × blend price × weight; blend = 85% input price + 15% output price
         % is after switch cost vs the warm model; raw ranking is not an approved switch
         now, sample count, input and output prices: --columns full
         switch requires at least 25% score improvement after switch cost, 3 consecutive passing checks and minimum warm time; ignored models cannot load

sources  discovery live, local scan  catalog live, 5 models, fetched 07:31:52  prices cached, fetched 07:30:00
         recovery monitoring, network requests reached this provider
```

`now` shows what the provider is doing. `next` lists its timers in clock order.
The score ladder puts the strongest raw scores first. Add `--columns full` for
current pressure, sample counts, and separate input/output prices.

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

The defaults check every minute, average up to 30 samples from the past 30 minutes, require
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

The `next` section shows both regular probe times and any pending post-switch
request alongside score checks and recovery timers. It refreshes after each
attempt, including failures and skips. Neither regular endpoint disappears when
an extra request is pending. For a model confirmed warm at 10:10 AM:

```text
next     10:11:00  check scores
         10:13:00  probe, production self-route after switch to Qwen3.5-9B
         10:30:00  probe, production self-route
         11:00:00  probe, local endpoint
```

Times use the timezone in the report header. Events on another day include the
date. An overdue turn is labeled `overdue`; its following regular turn is an
estimate until the first attempt runs or is skipped. Without a pending extra
request, `+3m after a switch is confirmed warm` describes the rule, not a scheduled
request. Dry runs show that no prompts are scheduled. `once` shows only a due
attempt it can make before exiting, with no recurring timeline.

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
5. Run a fresh first score check. Refresh the local model list, network pressure
   and input/output prices. Discard pre-shutdown pressure averages and passing-check
   counts, then start the **highest-scoring eligible model** using the same config
   and local endpoint settings. The winner may differ from the stopped model.
   Ignored, unavailable and `--model`-excluded models cannot win.
6. Confirm warm-up, start a new minimum warm-time clock, and resume normal score
   checks. Later switches use the usual improvement, consecutive-check and
   minimum warm-time requirements.
7. Check local inference and production self-routing three minutes after warm-up.
   Repeat those checks five minutes apart for up to 15 minutes. Score checks
   continue during this verification. If this provider receives network requests,
   recovery is confirmed. Otherwise leave it running and report recovery as
   unconfirmed.

Recovery probes ask for at most 16 output tokens and use a 120-second network
timeout per request. For a slower local model, set `--recovery-probe-timeout 180`
(allowed range: 30–300 seconds). Score checks can be delayed while a request is
waiting for a response. Hourly and post-switch probes retain their 30-second
timeout and 64-token limit.

If a local probe succeeds but Darkbloom still reports busy, recovery checks for
idle every 15 seconds, for up to 60 seconds. It keeps the successful result during
that wait and sends the production check once idle is confirmed. It does not send
another local prompt. The wait and deadline appear under `next`.
Score-based switching and regular probes pause during this short idle wait.

A timeout, failed check or provider change clears the failure count. The next
recovery attempt waits five minutes after the result, then waits for idle.
Regular and post-switch probes also wait through that five-minute pause. Busy
checks do not keep extending it by another five minutes. Restarting the manager
preserves the pause and any unfinished idle wait.

The output distinguishes local timeouts, HTTP errors, stale state, process/model
changes and reconnects. The last failed local and production results remain under
`sources` with their timestamps, even if the latest status becomes “busy.” Busy
can include a local request; it does not by itself prove that network traffic arrived.
Local timeouts never authorize a recovery shutdown.

The first recovery selection uses one fresh pressure sample with the usual
85% input / 15% output price blend and model weights. Nothing is warm at that
point, so there is no current model to protect with a score margin, switch cost,
passing-check wait or minimum warm time. Subsequent averages build normally.

If the local scan, network pressure or fresh prices are unavailable, or no model
is eligible, the provider stays offline. The manager retries the score check at
`--check-every`, without another stop command or another 15-minute wait. It does
not fall back to the old model or stale prices. A catalog outage alone does not
block a model whose local availability, pressure and prices were verified.

An ignored or removed old model does not prevent another eligible model from
winning. The selected model is checked again before launch. A changed config
or an external provider start cancels the saved restart. An interrupted or failed
launch is never repeated automatically; the selected target stays saved while
the manager checks whether it became warm.

Recovery has its own probes and leaves the hourly schedule intact. Regular and
post-switch probes stay paused through routing verification, then resume through
their usual one-at-a-time schedule. Score-based switching resumes as soon as
warm-up is confirmed. If a normal switch occurs during verification, that
verification ends without clearing the guard against repeated recovery shutdowns.

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

During the offline wait, the timeline shows when the fresh selection will run.
No restart model is promised before that check:

```text
now      STOPPED    LIVE, RECOVERY OFFLINE

next     10:34:00  refresh recovery status; score checks paused
         10:45:00  check fresh scores, then start the highest eligible model; 12m remaining after confirmed stop
         model switching and regular/after-switch probes paused during routing recovery

score    paused during routing recovery; no fresh ranking
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

The four gutter labels stay in the same order:

- `now`: warm model, time warm, request activity, session counts and the current decision.
- `next`: score checks, conditional switch time, recovery checks and probes, sorted
  by clock time. During a recovery shutdown it includes the saved restart deadline.
- `score`: models sorted by raw score, with the warm model marked `*`.
- `sources`: local discovery, catalog and price freshness, plus recovery status.

The session line shows requests served and tokens generated by this provider
since it started. It appears in both views and resets when the provider restarts.
The counts come from Darkbloom's local daemon state; no extra API key is needed.
Tokens are output tokens, matching Darkbloom's own session counter
([upstream implementation](https://github.com/Layr-Labs/d-inference/blob/master/provider-swift/Sources/ProviderCore/ProviderLoop%2BInferenceHandler.swift)).
These are provider-wide counts, not totals for just the warm model or the averaging
window. The manager does not accumulate them across restarts. A missing count
shows `N/A`; both counts show `N/A` when the provider is stopped or its state is stale.

In the example, Gemma leads by about 261% **after switch cost**, with two of
three checks passed. The third check is next. Gemma must still pass and the
provider must be idle before the manager can switch.

The ladder shows each exact model ID, its score, a bar relative to the highest
visible score, and its percentage above or below the warm model. Then come the
three score inputs: average pressure, blended price and weight. The raw score
and bar do not include switch cost; the percentage does. A percentage advantage
alone does not approve a switch.

`N/A` means a value is unavailable, not zero. Without one fresh warm model and
its score, there is no percentage comparison. When the warm score is zero,
`> zero` or `equal zero` replaces a percentage. Unknown scores sort last. Tied
scores keep their existing model order.

Warm means loaded in memory. Idle means the provider is not serving a request.
Pressure describes the network, not the number of requests reaching your Mac.
Prices show four decimals and scores show three; calculations use the full values.

For the detailed table, append `--columns full` to your command:

```sh
python3 warm_model_manager.py run --columns full
```

This example is a dry run. `--columns full` changes only the score section;
the timeline stays above it. `--columns ladder` selects the default again.
Neither choice resets samples, confirmations, saved timers or switching rules.

| Full table column | Meaning |
| --- | --- |
| `MODEL ID` | Exact ID accepted by `--ignore-model`, including capitalization and namespace. |
| `NOW` | Current public network pressure: active requests divided by warm providers, with a minimum denominator of 1. |
| `AVG 30m` | Average of retained pressure samples. The time label follows your check interval and sample limit. |
| `N` | Samples in that average. It can be below the limit after startup or during a data gap. |
| `IN$/M` | Input price per million input tokens. |
| `OUT$/M` | Output price per million output tokens. |
| `BLEND$/M` | Price per million total tokens at the fixed 85% input / 15% output mix, before pressure and weight. |
| `WEIGHT` | Model score multiplier, set with `--weight`. |
| `SCORE` | Average pressure × blended price × weight. |
| `STATUS` | Exclusion or missing-data reason. `IGNORED` is explicit; `AUTO-IGNORED` follows local availability or `--model`. |

Ignored and auto-ignored rows remain visible by default, including their math.
`Highest raw score` identifies the leader and any ties, with exclusions marked.
With `--hide-ignored`, it ranks only visible rows. Hidden models still have their
calculations saved, and excluded models can never be selected or loaded.

Terminal color distinguishes the warm model and the leading eligible score.
Redirected output stays plain text. Set `NO_COLOR=1` to disable color in a terminal.

| Decision | Meaning |
| --- | --- |
| `KEEP` | Keep the current model. |
| `WOULD SWITCH` | A dry run selected a different model; no launch occurs. |
| `SWITCH` | The manager plans to load a different model. The following log lines report the launch result. |
| `WARMING` | A launch was requested; the manager is waiting for Darkbloom to confirm the model is warm. |
| `DEFERRED` | A model was selected, but a condition such as active work or minimum warm time blocks the launch. |
| `WAIT` | There is no eligible scored target, or the current model's score is unavailable. |

The line under `now` explains the decision or shows the passing-check count.
An earliest switch in `next` is conditional: the candidate must keep passing,
the provider must be online and idle, and minimum warm time must have elapsed.
The estimate rounds up to the next score check after those requirements can
clear. A candidate below the required percentage has no switch countdown.
Pending warm-up has no invented completion time.

`next` reads the existing timers; it does not create extra checks or requests.
A slow network call or provider command can delay events. Source timestamps
show when catalog and price data were fetched; a stale cache stays labeled stale.

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
the manager is stopped. With the defaults of `60` and `30`, the average contains
at most 30 samples from the past 30 minutes. The table's `N` column shows how many are
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
| `--average-samples COUNT` | 30 | Maximum number of recent pressure samples to average. |
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
| Optional routing recovery | Stage, model/process identity, one network counter baseline, reconnect count, failure count, next check, fresh-selection retry time, previous/chosen model, stop/start intent, offline deadline, elapsed-time reference, config digest and paths, local endpoint launch flags, latest redacted check results and the one-attempt guard. No config contents or tokens. |

Each save replaces the latest snapshot and keeps a bounded sample window.

A recovery start clears the old pressure samples, passing-check counts and warm
residency before choosing from fresh data. It preserves the hourly probe schedule
and recovery attempt guard. The new warm-time clock starts only after confirmed
warm-up.

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
local availability. The `sources` line still says `discovery unavailable`.

### Why hasn't it switched?

Read the explanation under `now` and the `next` timeline. The candidate may need
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

Existing commands now show the timeline and score ladder. Add `--columns full`
to restore the detailed score table. `--hide-ignored` remains optional; it hides
excluded rows without changing their saved calculations or eligibility.
Choosing a display does not change saved state or switching rules.

The default average now uses up to 30 samples at one-minute intervals. If your
command includes `--average-samples 15` or `--history 15`, remove it or change it
to `30` to use the new default window. A custom `--check-every` still changes the
window length: interval × sample count.

On the first check with the new sample limit, the existing timing-change rule
clears the old averages and passing-check counts. The average fills as new
samples arrive; it does not wait 30 minutes before scoring. Minimum warm time,
pending launches and recovery state remain intact.

After this update, an existing recovery still in its offline wait will use fresh
scores when the wait ends. A launch already recorded as starting keeps its saved
target; upgrading does not issue a replacement launch. During saved verification,
normal score checks resume when running with `--apply --recover-routing`.

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
